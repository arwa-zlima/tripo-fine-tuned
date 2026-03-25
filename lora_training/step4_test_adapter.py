"""
Step 4 — Test a trained LoRA adapter with TripoSR.

Loads the base model, injects LoRA, loads adapter weights, runs inference
on a test image, and exports the resulting mesh.  Optionally compares
against the base model (no LoRA) using Chamfer distance.

Setup (Colab):
    # TripoSR repo must be on PYTHONPATH

Usage:
    python step4_test_adapter.py \
        --adapter adapters/lora_chair_bench.pt \
        --image test_images/chair.png \
        --output output/test_mesh.obj

    # With base-model comparison:
    python step4_test_adapter.py \
        --adapter adapters/lora_chair_bench.pt \
        --image test_images/chair.png \
        --output output/test_mesh.obj \
        --compare
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tsr.system import TSR  # noqa: E402
from tsr.models.transformer.lora import (  # noqa: E402
    load_lora_state_dict,
    set_lora_enabled,
)


def chamfer_distance(verts_a, verts_b, n_samples=10000):
    """Approximate Chamfer distance between two point clouds."""
    from scipy.spatial import cKDTree

    # Subsample if needed
    if len(verts_a) > n_samples:
        idx = np.random.choice(len(verts_a), n_samples, replace=False)
        verts_a = verts_a[idx]
    if len(verts_b) > n_samples:
        idx = np.random.choice(len(verts_b), n_samples, replace=False)
        verts_b = verts_b[idx]

    tree_a = cKDTree(verts_a)
    tree_b = cKDTree(verts_b)

    dist_a, _ = tree_b.query(verts_a)
    dist_b, _ = tree_a.query(verts_b)

    return float(np.mean(dist_a**2) + np.mean(dist_b**2))


def load_model(pretrained, device, lora_r=8, lora_alpha=16):
    """Load TripoSR and inject LoRA (without adapter weights)."""
    model = TSR.from_pretrained(pretrained, "config.yaml", "model.ckpt")
    model.to(device)
    model.eval()

    # Find and enable LoRA on backbone
    transformer = None
    for name, module in model.named_modules():
        if "Transformer1D" in type(module).__name__:
            transformer = module
            break
    if transformer is None:
        raise RuntimeError("Could not find Transformer1D backbone")

    transformer.enable_lora(r=lora_r, alpha=lora_alpha)
    return model


def run_inference(model, image_path, output_path, device, mc_resolution=256):
    """Run inference on a single image and export mesh."""
    image = Image.open(image_path).convert("RGB")

    t0 = time.time()
    with torch.no_grad():
        scene_codes = model(image, device)
    encode_time = time.time() - t0

    t0 = time.time()
    mesh = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)[0]
    mesh_time = time.time() - t0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.export(output_path)

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"  Mesh: {n_verts:,} verts, {n_faces:,} faces")
    print(f"  Encode: {encode_time:.2f}s, Mesh extraction: {mesh_time:.2f}s")
    print(f"  Saved: {output_path}")

    return mesh


def test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Image:   {args.image}")
    print(f"Adapter: {args.adapter}")

    # ── Load model with LoRA ──────────────────────────────────────────────
    print("\nLoading model...")
    model = load_model(args.pretrained, device,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # ── Load adapter weights ──────────────────────────────────────────────
    print("Loading adapter weights...")
    adapter_sd = torch.load(args.adapter, map_location="cpu", weights_only=True)
    load_lora_state_dict(model, adapter_sd)
    n_params = sum(v.numel() for v in adapter_sd.values())
    size_mb = os.path.getsize(args.adapter) / 1e6
    print(f"Adapter: {n_params:,} params, {size_mb:.1f} MB")

    # ── Run inference WITH adapter ────────────────────────────────────────
    print("\n--- Inference WITH LoRA adapter ---")
    mesh_lora = run_inference(model, args.image, args.output, device,
                              mc_resolution=args.mc_resolution)

    # ── Optionally compare with base model ────────────────────────────────
    if args.compare:
        print("\n--- Inference WITHOUT LoRA (base model) ---")
        set_lora_enabled(model, False)

        base_output = args.output.replace(".", "_base.")
        mesh_base = run_inference(model, args.image, base_output, device,
                                  mc_resolution=args.mc_resolution)

        # Re-enable LoRA
        set_lora_enabled(model, True)

        # Chamfer distance
        try:
            cd = chamfer_distance(mesh_lora.vertices, mesh_base.vertices)
            print(f"\nChamfer distance (LoRA vs base): {cd:.6f}")
            if cd < 1e-6:
                print("  → Adapters are effectively identical to base model (not yet trained?)")
            else:
                print(f"  → Adapter introduces measurable changes to the reconstruction")
        except ImportError:
            print("scipy not installed — skipping Chamfer distance (pip install scipy)")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Test a LoRA adapter with TripoSR")
    parser.add_argument("--adapter", required=True, help="Path to .pt adapter file")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="output/test_mesh.obj", help="Output mesh path")
    parser.add_argument("--pretrained", default="stabilityai/TripoSR",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--mc_resolution", type=int, default=256,
                        help="Marching cubes resolution")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--compare", action="store_true",
                        help="Also run base model and compute Chamfer distance")
    args = parser.parse_args()

    test(args)


if __name__ == "__main__":
    main()
