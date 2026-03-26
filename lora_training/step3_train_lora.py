"""
Step 3 — LoRA fine-tuning of the TripoSR transformer backbone.

Loads the pretrained TripoSR model, injects LoRA into the transformer,
and trains on multi-view rendered data produced by steps 1-2.

Only LoRA parameters are updated; the base model stays frozen.
The saved adapter is typically 5-20 MB (vs ~1.5 GB for the full model).

Setup (Colab):
    !pip install lpips tqdm
    # TripoSR repo must be on PYTHONPATH (or run from the repo root)

Usage:
    python step3_train_lora.py \
        --data_dir data/processed/chair/ \
        --output adapters/lora_chair_bench.pt \
        --epochs 10 \
        --lr 5e-5 \
        --render_size 128 \
        --lora_r 8 \
        --lora_alpha 16

The script prints progress every batch so Colab won't kill it for silence.
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure TripoSR is importable (works when running from lora_training/ or root)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tsr.system import TSR  # noqa: E402
from tsr.models.transformer.lora import get_lora_state_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Ray generation — single-view version of tsr.utils.get_spherical_cameras
# ---------------------------------------------------------------------------

def get_rays_for_view(elevation_deg, azimuth_deg, camera_distance, fovy_deg, height, width):
    """Generate (rays_o, rays_d) for one viewpoint using TSR's convention.

    Returns:
        rays_o: (H, W, 3)
        rays_d: (H, W, 3)
    """
    elev = torch.tensor([elevation_deg * math.pi / 180])
    azim = torch.tensor([azimuth_deg * math.pi / 180])
    dist = torch.tensor([camera_distance])

    cam_pos = torch.stack([
        dist * torch.cos(elev) * torch.cos(azim),
        dist * torch.cos(elev) * torch.sin(azim),
        dist * torch.sin(elev),
    ], dim=-1)  # (1, 3)

    center = torch.zeros_like(cam_pos)
    up = torch.tensor([[0.0, 0.0, 1.0]])
    fovy = torch.tensor([fovy_deg * math.pi / 180])

    lookat = F.normalize(center - cam_pos, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)

    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), cam_pos[:, :, None]], dim=-1
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0

    focal_length = 0.5 * height / torch.tan(0.5 * fovy)

    # Ray directions in camera space (focal=1, will divide by focal below)
    pixel_center = 0.5
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32) + pixel_center,
        torch.arange(height, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )
    directions = torch.stack(
        [(i - width / 2) / 1.0, -(j - height / 2) / 1.0, -torch.ones_like(i)], dim=-1
    )
    directions = F.normalize(directions, dim=-1)

    # Adjust for focal length
    directions = directions[None]  # (1, H, W, 3)
    directions[..., :2] = directions[..., :2] / focal_length[:, None, None, None]

    # Transform to world space
    rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
    rays_o = c2w[:, None, None, :3, 3].expand_as(rays_d)

    rays_d = F.normalize(rays_d, dim=-1)

    return rays_o[0], rays_d[0]  # (H, W, 3)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def parse_filename(fname):
    """Extract (distance, elevation, azimuth) from a step1-format filename.

    Format: image__distance_1_9__elevation_000__azimuth_045.png
    """
    base = os.path.splitext(fname)[0]
    parts = base.split("__")
    info = {}
    for part in parts:
        if part.startswith("distance_"):
            info["distance"] = float(part[len("distance_"):].replace("_", "."))
        elif part.startswith("elevation_"):
            info["elevation"] = float(part[len("elevation_"):])
        elif part.startswith("azimuth_"):
            info["azimuth"] = float(part[len("azimuth_"):])
    return info


class MultiViewDataset(torch.utils.data.Dataset):
    """Dataset that loads per-object multi-view renders.

    Each item returns one input view and one random target view from the
    same object, along with the target view's camera parameters.
    """

    def __init__(self, data_dir, image_size=512):
        self.image_size = image_size
        self.objects = []  # list of [(image_path, distance, elevation, azimuth), ...]

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        for obj_name in sorted(os.listdir(data_dir)):
            obj_dir = os.path.join(data_dir, obj_name)
            if not os.path.isdir(obj_dir):
                continue
            views = []
            for fname in sorted(os.listdir(obj_dir)):
                if not fname.lower().endswith(".png"):
                    continue
                info = parse_filename(fname)
                if "elevation" in info and "azimuth" in info:
                    views.append({
                        "path": os.path.join(obj_dir, fname),
                        "distance": info.get("distance", 1.9),
                        "elevation": info["elevation"],
                        "azimuth": info["azimuth"],
                    })
            if len(views) >= 2:
                self.objects.append(views)

        print(f"MultiViewDataset: {len(self.objects)} objects loaded from {data_dir}")

    def __len__(self):
        return len(self.objects)

    def _load_image(self, path):
        """Load an image as a float32 tensor in [0, 1], shape (H, W, 3)."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)

    def __getitem__(self, idx):
        views = self.objects[idx]

        # Pick a random input view and a different random target view
        indices = random.sample(range(len(views)), min(2, len(views)))
        input_view = views[indices[0]]
        target_view = views[indices[1]] if len(indices) > 1 else views[indices[0]]

        return {
            "input_image": self._load_image(input_view["path"]),
            "target_image": self._load_image(target_view["path"]),
            "target_elevation": target_view["elevation"],
            "target_azimuth": target_view["azimuth"],
            "target_distance": target_view["distance"],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load pretrained model ──────────────────────────────────────────
    print("Loading TripoSR model...")
    model = TSR.from_pretrained(
        args.pretrained, "config.yaml", "model.ckpt"
    )
    print("Model loaded.")

    # ── 2. Find Transformer1D backbone and enable LoRA ────────────────────
    transformer = None
    for name, module in model.named_modules():
        if "Transformer1D" in type(module).__name__:
            transformer = module
            print(f"Found transformer backbone at: {name}")
            break

    if transformer is None:
        print("ERROR: Could not find Transformer1D module in model.")
        sys.exit(1)

    # enable_lora() injects LoRA layers AND handles freezing internally:
    # it freezes all params in the transformer then re-enables lora_ params.
    # We call it BEFORE freezing the rest of the model so that the LoRA
    # Linear modules exist when we do the global freeze pass below.
    transformer.enable_lora(r=args.lora_r, alpha=args.lora_alpha,
                            dropout=args.lora_dropout)

    # Freeze every non-LoRA param across the entire model (image_tokenizer,
    # post_processor, decoder, renderer are all frozen — only LoRA trains).
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)

    # Move to device AFTER enabling LoRA so lora_A/lora_B tensors
    # are included in the GPU transfer.
    model.to(device)

    # Verify: count trainable vs total params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        print("ERROR: No trainable parameters found — LoRA injection failed.")
        sys.exit(1)
    print(f"LoRA enabled: training {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.3f}%)")

    # ── 3. Dataset & DataLoader ───────────────────────────────────────────
    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.isdir(train_dir):
        # Fallback: data_dir itself has per-object subdirectories (no split)
        print(f"No train/ subdir in {args.data_dir} — using root as train dir.")
        train_dir = args.data_dir

    val_dir = os.path.join(args.data_dir, "val")
    has_val = os.path.isdir(val_dir)

    dataset = MultiViewDataset(train_dir, image_size=args.image_size)
    if len(dataset) == 0:
        print(f"ERROR: No objects with >=2 views found in {train_dir}.")
        print("Check that step2_preprocess.py ran and produced per-object subdirs.")
        sys.exit(1)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )

    val_loader = None
    if has_val:
        val_dataset = MultiViewDataset(val_dir, image_size=args.image_size)
        if len(val_dataset) > 0:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, drop_last=False,
            )
            print(f"Val set: {len(val_dataset)} objects")

    # ── 4. Optimizer ──────────────────────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader), eta_min=args.lr * 0.1,
    )

    # ── 5. LPIPS loss (optional) ──────────────────────────────────────────
    loss_fn_lpips = None
    try:
        import lpips
        loss_fn_lpips = lpips.LPIPS(net="vgg").to(device).eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad_(False)
        print("LPIPS loss enabled.")
    except ImportError:
        print("lpips not installed — using MSE only. (pip install lpips)")

    # ── 6. Training loop ──────────────────────────────────────────────────
    model.train()
    # Keep renderer in eval mode (no randomized sampling)
    model.renderer.eval()

    fovy_deg = 40.0  # TSR default

    best_loss = float("inf")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            t0 = time.time()

            input_images = batch["input_image"]                      # (B, H, W, 3) CPU
            target_images = batch["target_image"].to(device)        # (B, H, W, 3)
            target_elev = batch["target_elevation"]                 # (B,) float
            target_azim = batch["target_azimuth"]                   # (B,) float
            target_dist = batch["target_distance"]                  # (B,) float

            B = input_images.shape[0]

            # TSR.forward() passes images through ImagePreprocessor, which
            # only recognises a 4-D batch when the tensor is exactly
            # torch.FloatTensor (CPU float32).  Passing a CUDA tensor or a
            # different dtype makes it fall into the "list" branch and adds
            # an extra batch dimension → 6-D tensor → einops crash.
            # Fix: convert each sample to a PIL Image so the preprocessor
            # handles them through its well-tested per-image path.
            pil_images = []
            for i in range(B):
                img_np = (input_images[i].numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))

            # Forward through TSR encoder
            scene_codes = model(pil_images, device)  # (B, 3, Cp, Hp, Wp)

            # Render from target viewpoints and accumulate loss
            loss = torch.tensor(0.0, device=device)

            for i in range(B):
                rays_o, rays_d = get_rays_for_view(
                    target_elev[i].item(),
                    target_azim[i].item(),
                    target_dist[i].item(),
                    fovy_deg,
                    args.render_size,
                    args.render_size,
                )
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)

                # TriplaneNeRFRenderer returns comp_rgb (H, W, 3) with white bg
                pred_rgb = model.renderer(
                    model.decoder, scene_codes[i], rays_o, rays_d
                )  # (H, W, 3)

                # Resize target to render_size for loss computation
                target_rgb = F.interpolate(
                    target_images[i:i+1].permute(0, 3, 1, 2),
                    (args.render_size, args.render_size),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)[0]  # (H, W, 3)

                # MSE loss
                loss_mse = F.mse_loss(pred_rgb, target_rgb)
                loss = loss + loss_mse

                # LPIPS loss
                if loss_fn_lpips is not None:
                    pred_lpips = pred_rgb.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    target_lpips = target_rgb.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                    loss = loss + loss_fn_lpips(pred_lpips, target_lpips).mean()

            loss = loss / B

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            dt = time.time() - t0
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", dt=f"{dt:.1f}s")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={avg_loss:.5f}", end="")

        # ── Validation loop ───────────────────────────────────────────────
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    v_input  = val_batch["input_image"]
                    v_target = val_batch["target_image"].to(device)
                    v_elev   = val_batch["target_elevation"]
                    v_azim   = val_batch["target_azimuth"]
                    v_dist   = val_batch["target_distance"]
                    vB = v_input.shape[0]
                    pil_val = [
                        Image.fromarray(
                            (v_input[i].numpy() * 255).astype(np.uint8)
                        )
                        for i in range(vB)
                    ]
                    v_codes = model(pil_val, device)
                    vl = torch.tensor(0.0, device=device)
                    for i in range(vB):
                        ro, rd = get_rays_for_view(
                            v_elev[i].item(), v_azim[i].item(),
                            v_dist[i].item(), fovy_deg,
                            args.render_size, args.render_size,
                        )
                        pred = model.renderer(
                            model.decoder, v_codes[i],
                            ro.to(device), rd.to(device),
                        )
                        gt = F.interpolate(
                            v_target[i:i+1].permute(0, 3, 1, 2),
                            (args.render_size, args.render_size),
                            mode="bilinear", align_corners=False,
                        ).permute(0, 2, 3, 1)[0]
                        vl = vl + F.mse_loss(pred, gt)
                    val_losses.append((vl / vB).item())
            val_loss = sum(val_losses) / len(val_losses)
            print(f"  val_loss={val_loss:.5f}", end="")
            model.train()
            model.renderer.eval()

        print()  # newline after epoch summary line

        # Save best adapter (use val loss if available, else train loss)
        metric = val_loss if val_loss is not None else avg_loss
        if metric < best_loss:
            best_loss = metric
            lora_sd = get_lora_state_dict(model)
            torch.save(lora_sd, args.output)
            size_mb = os.path.getsize(args.output) / 1e6
            print(f"  ✓ Saved best adapter: {args.output} ({size_mb:.1f} MB)")

    # Final save regardless
    lora_sd = get_lora_state_dict(model)
    final_path = args.output.replace(".pt", "_final.pt")
    torch.save(lora_sd, final_path)
    print(f"Final adapter saved: {final_path} ({os.path.getsize(final_path)/1e6:.1f} MB)")
    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of TripoSR")
    parser.add_argument("--data_dir", required=True,
                        help="Processed data directory (with train/ and val/ subdirs)")
    parser.add_argument("--output", default="adapters/lora_adapter.pt",
                        help="Output path for the LoRA adapter .pt file")
    parser.add_argument("--pretrained", default="stabilityai/TripoSR",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for Colab T4)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image resolution for TSR encoder")
    parser.add_argument("--render_size", type=int, default=128,
                        help="Rendering resolution for supervision (smaller = less VRAM)")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling = alpha/r)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Dropout on the LoRA path (0.0 = disabled)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
