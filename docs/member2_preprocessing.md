# Member 2 — Preprocessing & Data Loading (VerSe2020, T10–L5)

**Scope:** Member 2 only. These files are a reproducible research prototype for Module 1 segmentation, **not for clinical decision-making**. Member 1 inventory, validation reports and official splits remain unchanged.

## Implemented interfaces

| Component | Location | Purpose |
|---|---|---|
| NIfTI preprocessing | `src/module1_segmentation/data/preprocess.py` | Aligned CT/mask resampling, normalization, class mapping, reversible geometry metadata |
| PyTorch Dataset | `src/module1_segmentation/data/dataset.py` | Memory-mapped 3D patches and train/val DataLoaders |
| Data preparation CLI | `scripts/prepare_data.py` | Iterate Member 1 manifests, preserve splits, cache volumes, resume |
| DataLoader smoke test | `scripts/check_dataloader.py` | Inspect one real batch of images and masks |
| Cache validator | `scripts/validate_prepared_data.py` | Validate cached arrays and metadata, patient isolation |
| Settings | `configs/preprocessing.json` | Editable resolution, CT HU window, foreground-center sampling |
| Unit tests | `tests/module1/test_member2.py` | Labels, boundaries, reproducibility, data loaders; real NIfTI checks when nibabel available |

## Required ORIGINAL data (not present in the GitHub ZIP)

The CSVs in `outputs/m1_splits/` point to NIfTI files which are **not included** in the repository. You must obtain the original dataset under its terms and extract its three partitions preserving subfolders:

```text
data/raw/VerSe2020/
  dataset-01training/
    rawdata/sub-verse500/sub-verse500_dir-ax_ct.nii.gz
    derivatives/sub-verse500/sub-verse500_dir-ax_seg-vert_msk.nii.gz
  dataset-02validation/
    rawdata/...     derivatives/...
  dataset-03test/
    rawdata/...     derivatives/...
```

**Important:** Check the *exact* paths in the first row of `outputs/m1_splits/train.csv`; download packages might unpack into a differently named parent folder. Avoid committing `.nii.gz` or any original patient data to GitHub. Official VerSe dataset: https://github.com/anjany/verse and https://verse2020.grand-challenge.org/.

## Installation (Python 3.10+)

```powershell
python -m pip install -r requirements.txt
# PyTorch is separate to preserve your existing CPU/CUDA setup:
# https://pytorch.org/get-started/locally/
python -c "import torch, scipy, nibabel; print(torch.__version__)"
```

On Windows with the team's NVIDIA GPU, use the official PyTorch selector to install a compatible GPU build. CPU PyTorch is sufficient for DataLoader tests; GPU is only needed later for training.

## Run the pipeline from the repository root

```powershell
# 1. Confirm exact file paths and number of missing original volumes (no CT loaded)
python scripts/prepare_data.py --dry-run

# 2. After getting raw data, try a small real-data subset (one or two cases per split)
python scripts/prepare_data.py --max-cases 2

# 3. Check actual cached tensors and metadata
python scripts/validate_prepared_data.py --full
python scripts/check_dataloader.py --patch-size 96 96 96 --batch-size 2 --workers 0

# 4. Prepare the complete train and validation partitions (50 + 69 cases)
python scripts/prepare_data.py

# 5. Validate caches and run all tests
python scripts/validate_prepared_data.py --full
python -m unittest discover -s tests/module1 -p "test_*.py" -v

# 6. If agreed upon, prepare TEST separately only for final evaluation/inference
python scripts/prepare_data.py --splits test
```

The command is resumable and writes the prepared CSV after every successful case. Rerunning with unchanged parameters skips cached cases. To change preprocessing settings for existing cases, edit `configs/preprocessing.json` and use `--force`; **do not combine different configurations in one training run**. Disk space is needed for uncompressed `.npy` volumes; expect multiple GB depending on the original fields of view.

### Split policy

We DO NOT reshuffle patients, mix official split folders, or use test images for tuning. Member 1's original 179 eligible cases are 50 training / 69 validation / 60 test, with no patient overlap. The default pipeline processes only train and validation; test processing is opt-in. Validation and test use no augmentation.

### Geometry and intensities

- The CT and corresponding reference mask are required to have matching raw NIfTI shapes and affines. Misaligned inputs fail rather than being combined incorrectly.
- Both are resampled on the **exact same** RAS+ output grid at configurable voxel spacing. **Linear interpolation for CT; nearest-neighbor only for labels.** A missing target class after resampling is an error, not silently accepted.
- HU are clipped to the user-configured `[-300, 1500]` and linearly normalized to `[0,1]` for segmentation. This is a starting hyperparameter, **not a medically validated optimal setting**; original CT remains untouched for downstream HU analyses (Module 2).
- No segmentation-mask-based crop is used: **full physical field of view is retained**, so preprocessing can also be applied to unlabeled inference CTs using `preprocess_ct_only`.
- Saved arrays are in **ZYX** order while NIfTI shapes and affines are always **XYZ**. `metadata.json` includes original shape/affine, processed shape/affine, source paths, spacing, label mapping, and configuration fingerprint. Member 5 must restore predicted masks into original NIfTI world coordinates based on this metadata.

### VerSe labels and model contract

| Original mask label | Class passed to 3D U-Net | Level |
|---:|---:|---|
| All other labels (including original 0) | 0 | Background/non-target structures |
| 17 | 1 | T10 |
| 18 | 2 | T11 |
| 19 | 3 | T12 |
| 20 | 4 | L1 |
| 21 | 5 | L2 |
| 22 | 6 | L3 |
| 23 | 7 | L4 |
| 24 | 8 | L5 |

**Member 3 contract:** `in_channels=1`, `out_channels=9`, input `float32[B, 1, D, H, W]` normalized to `[0,1]`, target `int64[B, D, H, W]` with values 0–8. This is *semantic* 9-class segmentation rather than vertebra-vs-background binary segmentation.

**Member 4 example:**

```python
from src.module1_segmentation.data.dataset import build_dataloaders

train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
    "outputs/m2_preprocessing/train.csv",
    "outputs/m2_preprocessing/val.csv",
    patch_size_zyx=(96, 96, 96),  # D,H,W  -> confirm with M3 and GPU memory
    batch_size=2, num_workers=0, patches_per_case=12,
)
for epoch in range(epochs):
    train_dataset.set_epoch(epoch)  # deterministic but fresh train patch locations
    model.train()
    for batch in train_loader:
        images, targets = batch["image"].to(device), batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        # optimizer zero_grad -> backward -> step: Member 4 implements training
    # Validation contains exhaustive, deterministic, overlapping tiles of each CT.
    # For final case-level metrics, stitch logits/overlapping patches before scoring.
```

`persistent_workers=False` is intentional; otherwise `set_epoch` changes may not reach worker processes. Training patches are class-balanced by level, supplemented with random locations; left-right flip and mild intensity jitter are training-only. Validation patches cover all voxels including borders using a fixed sliding grid with overlap, without random sampling or augmentation. For *case-level* Dice, do **not** average per-patch Dice uncritically; reconstruct case-level probabilities or use Member 5's full-volume inference.

### On-disk output (never commit image data)

```text
data/processed/vertebrae_m2/training/<patient>__<scan>/
   image.npy         # float16 Z,Y,X normalized CT
   mask.npy          # uint8 Z,Y,X segmentation target 0..8
   centers.npy       # int32 [class,z,y,x] foreground candidates
   metadata.json     # geometry, source paths, preprocessing configuration
outputs/m2_preprocessing/train.csv
outputs/m2_preprocessing/val.csv
# test.csv appears only after explicit --splits test
```

These `.npy` files use NumPy memory mapping for CPU-side patch loading; each returned image patch is converted to PyTorch float32 and mask to int64. No CTs or masks need to be loaded together in training RAM.

## What is and is not yet verified

- Unit tests can run on synthetic `.npy` arrays without raw NIfTI files.
- The real NIfTI integration tests (including negative-axis affine orientation and resampling) require `nibabel` and are automatically **skipped** when it is not installed.
- **## Final Verification Results (2026-11-06)

- Training: 50/50 cases prepared and validated.
- Validation: 69/69 cases prepared and validated.
- Test: 60/60 cases prepared and validated.
- Total: 179/179 cases passed validation.
- Unit tests: 12/12 passed on Windows.
- Train/validation DataLoader smoke test: PASS.
- Target label coverage: classes 1-8 found in all three splits.
- No patient crossover detected.
- Preprocessing fingerprint: 72e39f20b4b0347c.

Pending: integration testing with Member 3's 3D U-Net.** until CT and segmentation NIfTI files are supplied. Hyperparameters (spacing, HU range, patch dimensions, foreground rate) should be optimized only on the training/validation splits in coordination with Members 3–4.

## Technical references

- Original dataset/labels: https://github.com/anjany/verse
- VerSe benchmark: https://doi.org/10.1016/j.media.2021.102166
- NiBabel world-geometry resampling: https://nipy.org/nibabel/reference/nibabel.processing.html
- PyTorch Dataset and DataLoader: https://docs.pytorch.org/docs/stable/data.html
