# TripoSR LoRA Training Toolkit

Train category-specific LoRA adapters for TripoSR.
Each adapter is ~5-20 MB and can be hot-swapped at runtime without reloading the 1.5 GB base model.

## Team Assignments

| Person | Categories           | Adapter filename              |
|--------|----------------------|-------------------------------|
| 1      | chair + bench        | `lora_chair_bench.pt`         |
| 2      | table + desk         | `lora_table_desk.pt`          |
| 3      | sofa + swivelchair   | `lora_sofa_swivelchair.pt`    |
| 4      | bed + wardrobe       | `lora_bed_wardrobe.pt`        |
| 5      | cabinet + bookshelf  | `lora_cabinet_bookshelf.pt`   |

---

## Path A — Google Colab

### 1. Setup Cell (run once)

```python
# Clone the repo
!git clone https://github.com/YOUR_ORG/TripoSR.git
%cd TripoSR

# Install dependencies
!pip install -r requirements.txt
!pip install -r lora_training/requirements.txt

# System dependencies for pyrender headless rendering
!apt-get install -y libosmesa6-dev freeglut3-dev

# Download your assigned .glb files from HuggingFace
# EDIT the --include pattern for YOUR categories
!huggingface-cli download mostafafaheem/furniture \
  --include "3D-FUTURE-model/chair/*" "3D-FUTURE-model/bench/*" \
  --local-dir data/furniture/ \
  --repo-type dataset
```

### 2. Render multi-view images (~5 min per 100 objects)

```python
# Render chair .glb files
!python lora_training/step1_render.py \
    --input_dir data/furniture/3D-FUTURE-model/chair/ \
    --output_dir data/rendered/chair/ \
    --resolution 512

# Render bench .glb files
!python lora_training/step1_render.py \
    --input_dir data/furniture/3D-FUTURE-model/bench/ \
    --output_dir data/rendered/bench/ \
    --resolution 512
```

### 3. Preprocess (~2 min per 100 objects)

```python
# For synthetic renders, rembg is NOT needed (alpha channel is clean)
!python lora_training/step2_preprocess.py \
    --input_dir data/rendered/chair/ \
    --output_dir data/processed/chair/

!python lora_training/step2_preprocess.py \
    --input_dir data/rendered/bench/ \
    --output_dir data/processed/bench/

# Merge into one training directory
!mkdir -p data/processed/chair_bench/train data/processed/chair_bench/val
!cp -r data/processed/chair/train/* data/processed/chair_bench/train/
!cp -r data/processed/chair/val/*   data/processed/chair_bench/val/
!cp -r data/processed/bench/train/* data/processed/chair_bench/train/
!cp -r data/processed/bench/val/*   data/processed/chair_bench/val/
```

### 4. Train LoRA (~30-60 min on T4, ~15-30 min on A100)

```python
!python lora_training/step3_train_lora.py \
    --data_dir data/processed/chair_bench/ \
    --output adapters/lora_chair_bench.pt \
    --epochs 10 \
    --lr 5e-5 \
    --batch_size 1 \
    --render_size 128 \
    --lora_r 8 \
    --lora_alpha 16
```

**Memory notes:**
- Batch size 1 + render_size 128 fits on T4 (16 GB)
- If you get OOM, try `--render_size 64`
- On A100 you can try `--batch_size 2 --render_size 192`

### 5. Test the adapter

```python
!python lora_training/step4_test_adapter.py \
    --adapter adapters/lora_chair_bench.pt \
    --image examples/chair.png \
    --output output/test_chair.obj \
    --compare
```

### 6. Download your adapter

```python
from google.colab import files
files.download('adapters/lora_chair_bench.pt')
```

---

## Path B — Kaggle

Same steps, with these differences:

### Setup

```python
!git clone https://github.com/YOUR_ORG/TripoSR.git
%cd TripoSR
!pip install -r requirements.txt
!pip install -r lora_training/requirements.txt
!apt-get install -y libosmesa6-dev freeglut3-dev
```

### Data download

Kaggle datasets can be attached via the sidebar, or:

```python
!pip install kagglehub
!huggingface-cli download mostafafaheem/furniture \
  --include "3D-FUTURE-model/chair/*" "3D-FUTURE-model/bench/*" \
  --local-dir /kaggle/working/data/furniture/ \
  --repo-type dataset
```

### GPU

- Enable GPU: Settings → Accelerator → GPU T4 x2
- Steps 2-5 are identical to Colab (adjust paths to `/kaggle/working/`)

### Download

Kaggle output files are in `/kaggle/working/`. Save the adapter:

```python
import shutil
shutil.copy('adapters/lora_chair_bench.pt', '/kaggle/working/lora_chair_bench.pt')
```

---

## Time Estimates (Colab T4)

| Step | 50 objects | 200 objects |
|------|-----------|-------------|
| step1 (render)     | ~3 min  | ~10 min  |
| step2 (preprocess) | ~1 min  | ~4 min   |
| step3 (train 10ep) | ~20 min | ~60 min  |
| step4 (test)       | ~30 sec | ~30 sec  |

---

## Adapter File Checklist

After training, verify your adapter:

```python
import torch, os

sd = torch.load('adapters/lora_chair_bench.pt', weights_only=True)
print(f"Keys:   {len(sd)}")
print(f"Params: {sum(v.numel() for v in sd.values()):,}")
print(f"Size:   {os.path.getsize('adapters/lora_chair_bench.pt') / 1e6:.1f} MB")

# Should print something like:
# Keys:   ~48-96 (depends on transformer depth)
# Params: ~500K - 2M
# Size:   2-10 MB
```

If the file is >50 MB, something went wrong (you may have saved full model weights).

---

## Troubleshooting

**OOM on T4**: Reduce `--render_size 64` or `--batch_size 1`

**pyrender import error**: Make sure `libosmesa6-dev` is installed and
`PYOPENGL_PLATFORM=osmesa` is set BEFORE importing pyrender

**Adapter has no effect**: Check that `step4_test_adapter.py --compare` shows
non-zero Chamfer distance. If CD ≈ 0, the adapter wasn't trained properly.

**Colab disconnects**: The training script prints every batch. If you still get
disconnected, add `--epochs 3` for shorter runs and resume manually.
