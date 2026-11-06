"""Member 2: make resumable, patient-safe VerSe2020 segmentation volumes.

From repository root:
    python scripts/prepare_data.py --dry-run
    python scripts/prepare_data.py --max-cases 2
    python scripts/prepare_data.py

Default processes official training and validation splits ONLY. Test is opt-in.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module1_segmentation.data.preprocess import (  # noqa: E402
    PreprocessSettings, portable_path, resolve_path, write_case,
)

SPLITS = {"training": "train.csv", "validation": "val.csv", "test": "test.csv"}
OUTPUT_FIELDS = ["dataset_split", "patient_id", "scan_id", "image_path", "mask_path",
                 "centers_path", "metadata_path", "shape_zyx", "classes_present", "config_fingerprint"]


def read_m1_manifests(folder: Path) -> dict[str, list[dict[str, str]]]:
    all_cases = {}
    patient_split = {}
    case_ids = set()
    for split, fname in SPLITS.items():
        path = folder / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing Member 1 split file: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"dataset_split", "patient_id", "scan_id", "ct_path", "mask_path", "target_count", "validation_status"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Invalid Member 1 CSV: {path}; missing {required - set(reader.fieldnames or [])}")
            rows = list(reader)
        for row in rows:
            patient, scan = row["patient_id"], row["scan_id"]
            if row["dataset_split"] != split:
                raise ValueError(f"Incorrect split assignment for {scan}")
            if (split, patient, scan) in case_ids:
                raise ValueError(f"Duplicate case {split}/{patient}/{scan}")
            case_ids.add((split, patient, scan))
            old = patient_split.setdefault(patient, split)
            if old != split:
                raise ValueError(f"Patient {patient} occurs in {old} and {split} splits")
            if row["validation_status"] != "passed" or not (1 <= int(row["target_count"]) <= 8):
                raise ValueError(f"Non-eligible case in Member 1 split CSV: {split}/{scan}")
        all_cases[split] = rows
    return all_cases


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8-sig", dir=path.parent,
                                     prefix=".m2_", suffix=".tmp", delete=False) as f:
        staging = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
    os.replace(staging, path)


def find_prepared(rows: list[dict[str, str]], processed_dir: Path, settings: PreprocessSettings) -> list[dict[str, str]]:
    """Inventory cached cases without loading a single NIfTI or .npy volume."""
    import re
    result = []
    for row in rows:
        patient, scan = row["patient_id"], row["scan_id"]
        if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) for name in (patient, scan)):
            raise ValueError(f"Unsafe case ID: {patient} {scan}")
        folder = processed_dir / row["dataset_split"] / f"{patient}__{scan}"
        required = [folder / name for name in ("image.npy", "mask.npy", "centers.npy", "metadata.json")]
        if not all(f.is_file() for f in required):
            continue
        with (folder / "metadata.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("config_fingerprint") != settings.fingerprint():
            continue
        if meta.get("dataset_split") != row["dataset_split"] or meta.get("patient_id") != patient or meta.get("scan_id") != scan:
            raise ValueError(f"Cached case identity mismatch: {folder}")
        for key, source_col in (("source_ct", "ct_path"), ("source_mask", "mask_path")):
            expected = portable_path(resolve_path(row[source_col]))
            if meta.get(key) != expected:
                raise ValueError(f"Cached source path differs from Member 1 CSV: {folder}; use --force")
        result.append({
            "dataset_split": row["dataset_split"], "patient_id": patient, "scan_id": scan,
            "image_path": portable_path(folder / "image.npy"),
            "mask_path": portable_path(folder / "mask.npy"),
            "centers_path": portable_path(folder / "centers.npy"),
            "metadata_path": portable_path(folder / "metadata.json"),
            "shape_zyx": json.dumps(meta["processed_shape_zyx"]),
            "classes_present": json.dumps(meta["target_classes_present"]),
            "config_fingerprint": meta["config_fingerprint"],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=ROOT / "outputs/m1_splits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/vertebrae_m2")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "outputs/m2_preprocessing")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/preprocessing.json")
    parser.add_argument("--spacing", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["training", "validation"])
    parser.add_argument("--max-cases", type=int, default=None, help="Per-split cap; useful for the first smoke test")
    parser.add_argument("--force", action="store_true", help="Recompute selected cases even if already prepared")
    parser.add_argument("--dry-run", action="store_true", help="Report required dataset paths without loading images")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be >= 1")
    with args.config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if args.spacing is not None:
        config["spacing_xyz_mm"] = args.spacing
    if "spacing_xyz_mm" in config:
        config["spacing_xyz_mm"] = tuple(config["spacing_xyz_mm"])
    settings = PreprocessSettings(**config)
    rows_by_split = read_m1_manifests(args.split_dir)
    selected = {split: rows_by_split[split][:args.max_cases] if args.max_cases is not None else rows_by_split[split]
                for split in dict.fromkeys(args.splits)}
    missing = []
    already_prepared = 0
    for split, rows in selected.items():
        prepared = find_prepared(rows, args.output_dir, settings)
        already_prepared += len(prepared)
        prepped_ids = {(r["patient_id"], r["scan_id"]) for r in prepared}
        for row in rows:
            if not args.force and (row["patient_id"], row["scan_id"]) in prepped_ids:
                continue
            ct, mask = resolve_path(row["ct_path"]), resolve_path(row["mask_path"])
            if not ct.is_file() or not mask.is_file():
                missing.append((row["scan_id"], ct, mask))
    print(f"M1 official eligible counts: { {s: len(r) for s,r in rows_by_split.items()} }")
    print(f"Requested: { {s: len(r) for s,r in selected.items()} }; preprocessed already: {already_prepared}")
    print(f"Config: spacing XYZ={settings.spacing_xyz_mm} mm; HU [{settings.hu_min},{settings.hu_max}]; "
          f"fingerprint={settings.fingerprint()}")
    print(f"Missing input pairs: {len(missing)}")
    for scan, ct, mask in missing[:3]:
        print(f"  {scan}\n    CT: {ct}\n    Mask: {mask}")
    if args.dry_run:
        return
    if missing:
        parser.error("Original VerSe2020 volumes are not available. Download/extract them according to the root README "
                     "and preserve the paths in Member 1's CSVs. Do not regenerate or reshuffle splits.")
    for split, rows in selected.items():
        for index, row in enumerate(rows, 1):
            result = write_case(row, args.output_dir, settings, overwrite=args.force)
            print(f"[{split} {index}/{len(rows)}] {result['scan_id']}  shape={result['shape_zyx']}", flush=True)
            # Commit progress after each case to make interrupted runs resumable.
            atomic_csv(args.manifest_dir / SPLITS[split], find_prepared(rows_by_split[split], args.output_dir, settings))
    summary = {s: len(find_prepared(rows_by_split[s], args.output_dir, settings)) for s in SPLITS}
    for split, rows in rows_by_split.items():
        if summary[split] > 0:
            atomic_csv(args.manifest_dir / SPLITS[split], find_prepared(rows, args.output_dir, settings))
    print("Prepared cases:", summary)
    print("Manifests:", args.manifest_dir.resolve())
    print("No training/test patient leakage: official partitions unchanged.")


if __name__ == "__main__":
    main()
