"""Member 2: validate cached processed .npy files without raw NIfTI data.

Run after scripts/prepare_data.py (or on a different machine with copied caches).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module1_segmentation.data.dataset import load_manifest  # noqa: E402
from src.module1_segmentation.data.preprocess import resolve_path  # noqa: E402

NAMES = {"training": "train.csv", "validation": "val.csv", "test": "test.csv"}


def validate_case(row: dict[str, str], full: bool = False) -> None:
    img = np.load(resolve_path(row["image_path"]), mmap_mode="r", allow_pickle=False)
    mask = np.load(resolve_path(row["mask_path"]), mmap_mode="r", allow_pickle=False)
    centers = np.load(resolve_path(row["centers_path"]), allow_pickle=False)
    with resolve_path(row["metadata_path"]).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    case = row["scan_id"]
    if img.ndim != 3 or img.shape != mask.shape or img.dtype != np.float16 or mask.dtype != np.uint8:
        raise ValueError(f"Incorrect image/mask shape or dtype: {case}")
    if list(img.shape) != meta["processed_shape_zyx"] or meta["array_order"] != "ZYX":
        raise ValueError(f"NIfTI/NumPy axis ordering mismatch: {case}")
    if meta["config_fingerprint"] != row["config_fingerprint"]:
        raise ValueError(f"Preprocessing fingerprint mismatch: {case}")
    if row["patient_id"] != meta["patient_id"] or row["dataset_split"] != meta["dataset_split"]:
        raise ValueError(f"Metadata identity mismatch: {case}")
    if centers.ndim != 2 or centers.shape[1] != 4 or not (1 <= len(centers)):
        raise ValueError(f"Missing/invalid positive sample centers: {case}")
    if np.any((centers[:, 0] < 1) | (centers[:, 0] > 8)):
        raise ValueError(f"Out-of-range center label: {case}")
    if np.any(centers[:, 1:] < 0) or np.any(centers[:, 1:] >= np.asarray(img.shape)[None, :]):
        raise ValueError(f"Out-of-volume center coordinate: {case}")
    found_labels = mask[centers[:, 1], centers[:, 2], centers[:, 3]]
    if not np.array_equal(found_labels, centers[:, 0]):
        raise ValueError(f"Saved center/segmentation mismatch: {case}")
    aff = np.asarray(meta["processed_affine_xyz"], dtype=float)
    raw = np.asarray(meta["raw_affine_xyz"], dtype=float)
    if aff.shape != (4, 4) or raw.shape != (4, 4) or not np.isfinite(aff).all() or not np.isfinite(raw).all():
        raise ValueError(f"Invalid affine: {case}")
    if not np.allclose(aff[3], [0, 0, 0, 1]) or abs(np.linalg.det(aff[:3, :3])) < 1e-9:
        raise ValueError(f"Invalid processed coordinate system: {case}")
    # Inspect a sparse volume subset by default; --full scans each slice.
    step = 1 if full else max(1, img.shape[0] // 7)
    for z in range(0, img.shape[0], step):
        slab = img[z]
        if not np.isfinite(slab).all() or np.any((slab < 0) | (slab > 1)):
            raise ValueError(f"Invalid normalized intensities: {case}, slice={z}")
        if np.any(mask[z] > 8):
            raise ValueError(f"Invalid target class: {case}, slice={z}")
    if full:
        # Labels in a full-volume must agree with metadata; not just cached centers.
        actual = sorted(int(v) for v in np.unique(mask) if v > 0)
        if actual != sorted(meta["target_classes_present"]):
            raise ValueError(f"Target classes differ from preprocessing metadata: {case}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=ROOT / "outputs/m2_preprocessing")
    p.add_argument("--full", action="store_true", help="Scan all voxels/slices instead of a sample")
    args = p.parse_args()
    patient_owner: dict[str, str] = {}
    fingerprints = set()
    all_case_keys = set()
    counts = {}
    for split, csvfile in NAMES.items():
        path = args.manifest_dir / csvfile
        if not path.exists():
            print(f"No prepared {split} manifest (allowed): {path}")
            continue
        rows = load_manifest(path)
        for row in rows:
            patient = row["patient_id"]
            previous = patient_owner.setdefault(patient, split)
            if previous != split:
                raise ValueError(f"Patient leakage: {patient}: {previous} and {split}")
            key = (split, patient, row["scan_id"])
            if key in all_case_keys:
                raise ValueError(f"Duplicate case {key}")
            all_case_keys.add(key)
            if row["dataset_split"] != split:
                raise ValueError(f"Case in wrong split: {row['scan_id']}")
            fingerprints.add(row["config_fingerprint"])
            validate_case(row, full=args.full)
        counts[split] = len(rows)
        print(f"PASS {split}: {len(rows)} prepared cases", flush=True)
    if not counts:
        raise FileNotFoundError("No Member 2 processed CSV manifests found; run scripts/prepare_data.py")
    if len(fingerprints) != 1:
        raise ValueError("Mixed preprocessing configs across partitions")
    print(f"PASS: {sum(counts.values())} prepared cases, no patient crossover, "
          f"consistent fingerprint={next(iter(fingerprints))}")


if __name__ == "__main__":
    main()
