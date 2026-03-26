"""
Step 2 — Preprocess rendered images: background removal + train/val split.

For synthetic renders from step1 the alpha channel is already clean,
so rembg is skipped by default.  Use --use_rembg for real photographs.

Setup (Colab):
    !pip install rembg Pillow numpy

Usage:
    python step2_preprocess.py \
        --input_dir data/rendered/chair/ \
        --output_dir data/processed/chair/ \
        --val_ratio 0.2
"""

import argparse
import os
import random
import shutil

import numpy as np
from PIL import Image


def composite_on_white(image_path, output_path, size=512):
    """Load an RGBA image and composite onto a white background."""
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((size, size), Image.LANCZOS)

    # Composite RGBA onto white
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(background, img)
    composite.convert("RGB").save(output_path)


def process_with_rembg(image_path, output_path, rembg_session, foreground_ratio=0.85, size=512):
    """Remove background with rembg, recenter, composite onto white."""
    import rembg

    img = Image.open(image_path).convert("RGBA")

    # Only run rembg if the alpha channel looks fully opaque (= real photo)
    extrema = img.getextrema()
    if extrema[3][0] == 255:
        img = rembg.remove(img, session=rembg_session)

    img_np = np.array(img)

    # Crop to foreground bounding box
    alpha = img_np[..., 3]
    if alpha.max() == 0:
        # Fully transparent — just save white image
        Image.new("RGB", (size, size), (255, 255, 255)).save(output_path)
        return

    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    fg = img_np[y1:y2+1, x1:x2+1]

    # Pad to square
    h, w = fg.shape[:2]
    side = max(h, w)
    ph = (side - h) // 2
    pw = (side - w) // 2
    fg = np.pad(fg, ((ph, side - h - ph), (pw, side - w - pw), (0, 0)),
                mode="constant", constant_values=0)

    # Pad for foreground ratio
    new_size = int(fg.shape[0] / foreground_ratio)
    pad = (new_size - fg.shape[0]) // 2
    fg = np.pad(fg, ((pad, new_size - fg.shape[0] - pad),
                      (pad, new_size - fg.shape[0] - pad), (0, 0)),
                mode="constant", constant_values=0)

    # Composite onto white
    fg_img = Image.fromarray(fg).resize((size, size), Image.LANCZOS)
    background = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    composite = Image.alpha_composite(background, fg_img)
    composite.convert("RGB").save(output_path)


def split_and_process(input_dir, output_dir, val_ratio, use_rembg, image_size):
    """Organize objects into train/val splits with processed images."""
    # Each subdirectory in input_dir is one object
    objects = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    if not objects:
        print(f"No object subdirectories found in {input_dir}")
        return

    # Shuffle deterministically for reproducibility
    rng = random.Random(42)
    rng.shuffle(objects)

    n_val = max(1, int(len(objects) * val_ratio))
    val_objects = set(objects[:n_val])
    train_objects = set(objects[n_val:])

    print(f"Objects: {len(objects)} total, {len(train_objects)} train, {len(val_objects)} val")

    rembg_session = None
    if use_rembg:
        import rembg
        rembg_session = rembg.new_session()

    total_images = 0
    for split_name, split_objects in [("train", train_objects), ("val", val_objects)]:
        for obj_name in sorted(split_objects):
            src_dir = os.path.join(input_dir, obj_name)
            dst_dir = os.path.join(output_dir, split_name, obj_name)
            os.makedirs(dst_dir, exist_ok=True)

            image_files = sorted([
                f for f in os.listdir(src_dir)
                if f.lower().endswith(".png")
            ])

            if not image_files:
                print(f"  [{split_name}] {obj_name}: WARNING — no PNG files found, skipping")
                continue

            for img_file in image_files:
                src_path = os.path.join(src_dir, img_file)
                dst_path = os.path.join(dst_dir, img_file)

                if use_rembg:
                    process_with_rembg(src_path, dst_path, rembg_session,
                                       size=image_size)
                else:
                    composite_on_white(src_path, dst_path, size=image_size)

            total_images += len(image_files)
            print(f"  [{split_name}] {obj_name}: {len(image_files)} images")

    print(f"Done. Total images processed: {total_images}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess rendered images")
    parser.add_argument("--input_dir", required=True,
                        help="Directory with per-object subdirectories from step1")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (will contain train/ and val/ subdirs)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of objects for validation (default: 0.2)")
    parser.add_argument("--use_rembg", action="store_true",
                        help="Run rembg background removal (for real photos, not needed for step1 renders)")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Output image resolution (default: 512)")
    args = parser.parse_args()

    split_and_process(
        args.input_dir, args.output_dir,
        args.val_ratio, args.use_rembg, args.image_size,
    )


if __name__ == "__main__":
    main()
