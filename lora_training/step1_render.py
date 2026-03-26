"""
Step 1 — Render multi-view images from .glb mesh files.

Uses pyrender + trimesh for headless GPU-free rendering.
Camera poses match TSR's coordinate convention (z-up) exactly,
so the rendered images can be used directly for LoRA training.

Setup (Colab):
    !apt-get install -y libosmesa6-dev freeglut3-dev
    !pip install pyrender trimesh Pillow numpy

Usage:
    python step1_render.py \
        --input_dir data/glb/chair/ \
        --output_dir data/rendered/chair/ \
        --resolution 512
"""

import argparse
import math
import os

import numpy as np
import trimesh
from PIL import Image

# Must be set BEFORE importing pyrender.
# Use EGL (GPU-accelerated headless) on Colab/Linux with a GPU.
# Fall back to osmesa (software) only if EGL is unavailable.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402


# ---------------------------------------------------------------------------
# Camera math — matches tsr.utils.get_spherical_cameras exactly
# ---------------------------------------------------------------------------

def get_camera_pose(elevation_deg, azimuth_deg, distance):
    """Compute a 4x4 camera-to-world matrix using TSR's convention (z-up).

    This replicates the exact math in ``tsr.utils.get_spherical_cameras``
    so that rays generated at training time will match these renders.
    """
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)

    # Camera position in z-up world coordinates
    x = distance * math.cos(elev) * math.cos(azim)
    y = distance * math.cos(elev) * math.sin(azim)
    z = distance * math.sin(elev)
    cam_pos = np.array([x, y, z])

    # Forward = look toward origin
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)

    # World up = +z
    world_up = np.array([0.0, 0.0, 1.0])

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera is at the pole — pick an arbitrary right vector
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    # OpenGL camera convention: columns = [right, up, -forward, position]
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward  # camera looks along -z
    c2w[:3, 3] = cam_pos
    return c2w


# ---------------------------------------------------------------------------
# Mesh loading & normalization
# ---------------------------------------------------------------------------

def load_and_normalize_mesh(mesh_path):
    """Load a .glb mesh, convert y-up → z-up, center, and scale to unit box."""
    mesh = trimesh.load(mesh_path, force="mesh")

    # glTF uses y-up; TSR uses z-up.  Rotate -90° around x.
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rot)

    # Center at origin
    centroid = mesh.bounding_box.centroid
    mesh.apply_translation(-centroid)

    # Scale so the longest axis fits in [-0.5, 0.5]
    extent = mesh.bounding_box.extents.max()
    if extent > 0:
        mesh.apply_scale(1.0 / extent)

    return mesh


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_mesh(mesh, elevation_deg, azimuth_deg, distance, fovy_deg, resolution):
    """Render a single RGBA image from the given viewpoint."""
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.6, 0.6, 0.6])

    # Convert trimesh → pyrender mesh
    # Use vertex colors if available, otherwise flat gray
    if mesh.visual.kind == "vertex" and hasattr(mesh.visual, "vertex_colors"):
        py_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.7, 0.7, 0.7, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.8,
        )
        py_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene.add(py_mesh)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=math.radians(fovy_deg))
    cam_pose = get_camera_pose(elevation_deg, azimuth_deg, distance)
    scene.add(camera, pose=cam_pose)

    # Light that follows the camera
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    return color  # uint8 RGBA (H, W, 4)


def render_all_views(
    mesh_path,
    output_dir,
    distance=1.9,
    fovy_deg=40.0,
    resolution=512,
    n_azimuths=8,
    elevations=(0, 20),
):
    """Render multi-view images for one mesh and save with pose filenames."""
    mesh = load_and_normalize_mesh(mesh_path)
    obj_name = os.path.splitext(os.path.basename(mesh_path))[0]
    obj_out = os.path.join(output_dir, obj_name)
    os.makedirs(obj_out, exist_ok=True)

    azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
    dist_str = str(distance).replace(".", "_")

    count = 0
    for elev in elevations:
        for azim in azimuths:
            rgba = render_mesh(mesh, elev, azim, distance, fovy_deg, resolution)
            fname = f"image__distance_{dist_str}__elevation_{int(elev):03d}__azimuth_{int(azim):03d}.png"
            Image.fromarray(rgba).save(os.path.join(obj_out, fname))
            count += 1

    print(f"  [{obj_name}] rendered {count} views → {obj_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render multi-view images from .glb files")
    parser.add_argument("--input_dir", required=True, help="Directory containing .glb files")
    parser.add_argument("--output_dir", required=True, help="Output directory for rendered images")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--distance", type=float, default=1.9, help="Camera distance (TSR default: 1.9)")
    parser.add_argument("--fovy", type=float, default=40.0, help="Vertical FOV in degrees (TSR default: 40)")
    parser.add_argument("--n_azimuths", type=int, default=8, help="Number of azimuth angles")
    parser.add_argument("--elevations", type=float, nargs="+", default=[0, 20],
                        help="Elevation angles in degrees")
    args = parser.parse_args()

    glb_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.lower().endswith(".glb")]
    )
    if not glb_files:
        print(f"No .glb files found in {args.input_dir}")
        return

    print(f"Found {len(glb_files)} .glb files in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    ok, failed = 0, []
    for i, fname in enumerate(glb_files):
        print(f"[{i+1}/{len(glb_files)}] {fname}")
        try:
            render_all_views(
                os.path.join(args.input_dir, fname),
                args.output_dir,
                distance=args.distance,
                fovy_deg=args.fovy,
                resolution=args.resolution,
                n_azimuths=args.n_azimuths,
                elevations=args.elevations,
            )
            ok += 1
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed.append(fname)

    print(f"\nDone. {ok}/{len(glb_files)} rendered successfully.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
