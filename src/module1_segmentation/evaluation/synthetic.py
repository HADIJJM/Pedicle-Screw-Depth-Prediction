"""Manufactured VerSe-like NIfTI cases for plumbing tests (NOT medical data).

Five cases exercise every evaluation path:
  syn-perfect   prediction identical to the reference
  syn-shift     every vertebra named one level too caudally (best shift -1)
  syn-missfp    LPS-oriented scan, L4 missed, a false-positive "T10" blob, T10/T11 absent
  syn-pipeline   LPS scan sent through preprocessing and the inference save step
  syn-offgrid   prediction left on the 2 mm processed grid (evaluator must resample)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.module1_segmentation.evaluation.case_io import map_verse_labels

SHAPE = (56, 52, 128)
SPACING = (1.2, 1.2, 2.5)


def _phantom(verse_labels, shape=SHAPE, seed=0):
    """RAS voxel grid: a soft-tissue cylinder with vertebral bodies stacked along S."""
    rng = np.random.default_rng(seed)
    ct = np.full(shape, -1000.0, dtype=np.float32)
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    body = (xx - shape[0] / 2) ** 2 + (yy - shape[1] / 2) ** 2 < (0.45 * min(shape[:2])) ** 2
    ct[body, :] = 40.0
    mask = np.zeros(shape, dtype=np.uint8)
    slab = shape[2] // (len(verse_labels) + 1)
    for i, lab in enumerate(verse_labels):  # VerSe labels increase caudally -> top first
        z1 = shape[2] - 4 - i * slab
        z0 = z1 - slab + 3
        mask[18:38, 16:34, z0:z1] = lab
    ct[mask > 0] = 650.0
    ct += rng.normal(0, 25, size=shape).astype(np.float32)
    return ct, mask


def _affine(shape=SHAPE, spacing=SPACING, lps=False):
    aff = np.diag([*spacing, 1.0])
    aff[:3, 3] = [-shape[0] * spacing[0] / 2, -shape[1] * spacing[1] / 2, -200.0]
    if lps:  # same world geometry, voxel axes x and y reversed
        flip = np.diag([-1.0, -1.0, 1.0, 1.0])
        flip[0, 3], flip[1, 3] = shape[0] - 1, shape[1] - 1
        aff = aff @ flip
    return aff


def _to_lps(arr):
    return np.ascontiguousarray(arr[::-1, ::-1, :])


def make_synthetic_split(root: Path):
    """Write CT/masks/predictions under root; returns (manifest_path, prediction_dir)."""
    import nibabel as nib
    from src.module1_segmentation.data.preprocess import PreprocessSettings, preprocess_pair
    from src.module1_segmentation.inference.save_output import save_mask_two_spaces

    root = Path(root)
    raw, pred_dir = root / "data/raw/synthetic", root / "preds"
    raw.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def write_case(scan, ct, mask, aff):
        ct_p, m_p = raw / f"{scan}_ct.nii.gz", raw / f"{scan}_seg-vert_msk.nii.gz"
        nib.save(nib.Nifti1Image(ct.astype(np.int16), aff), str(ct_p))
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), str(m_p))
        targets = int(np.isin(np.unique(mask), np.arange(17, 25)).sum())
        rows.append({"dataset_split": "validation", "patient_id": scan, "scan_id": scan,
                     "ct_path": ct_p.relative_to(root).as_posix(), "mask_path": m_p.relative_to(root).as_posix(),
                     "centroid_path": "", "coverage_type": "full" if targets == 8 else "partial",
                     "target_count": targets, "validation_status": "passed"})
        return ct_p, m_p

    def save_pred(scan, classes, aff):
        nib.save(nib.Nifti1Image(classes.astype(np.uint8), aff), str(pred_dir / f"{scan}_pred.nii.gz"))

    full = list(range(15, 26))  # T8..L6: includes non-target labels on both ends
    aff = _affine()

    ct, mask = _phantom(full, seed=1)
    write_case("syn-perfect", ct, mask, aff)
    save_pred("syn-perfect", map_verse_labels(mask), aff)

    ct, mask = _phantom(full, seed=2)
    write_case("syn-shift", ct, mask, aff)
    shifted = np.where(mask > 0, mask + 1, 0).astype(np.uint8)  # true T9 called "T10", ... L5 called "L6"
    save_pred("syn-shift", map_verse_labels(shifted), aff)

    ct, mask = _phantom(list(range(19, 26)), seed=3)  # T12..L6 only: T10/T11 absent
    ct, mask, aff_lps = _to_lps(ct), _to_lps(mask), _affine(lps=True)
    write_case("syn-missfp", ct, mask, aff_lps)
    pred = map_verse_labels(mask)
    pred[pred == 7] = 0          # L4 missed
    pred[2:6, 2:6, 100:106] = 1  # false-positive "T10" blob
    save_pred("syn-missfp", pred, aff_lps)

    ct, mask = _phantom(full, seed=4)
    ct, mask = _to_lps(ct), _to_lps(mask)
    ct_p, m_p = write_case("syn-pipeline", ct, mask, aff_lps)
    _, mask_zyx, meta = preprocess_pair(ct_p, m_p, PreprocessSettings())
    save_mask_two_spaces(mask_zyx, meta, output_original=pred_dir / "syn-pipeline_pred.nii.gz")

    ct, mask = _phantom(full, seed=5)
    ct_p, m_p = write_case("syn-offgrid", ct, mask, aff)
    _, mask_zyx, meta = preprocess_pair(ct_p, m_p, PreprocessSettings())
    save_mask_two_spaces(mask_zyx, meta, output_original=root / "unused_original.nii.gz",
                         output_processed=pred_dir / "syn-offgrid_pred.nii.gz")

    manifest = root / "val.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:  # BOM like the split manifests
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return manifest, pred_dir
