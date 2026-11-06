"""Build reproducible, leakage-free VerSe2020 manifests from existing reports.

No CT volumes are loaded, no random re-splitting is performed.
Run from anywhere: python src/module1_segmentation/data/split.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

EXPECTED = {"training": 61, "validation": 80, "test": 73}
NAMES = {"training": "train.csv", "validation": "val.csv", "test": "test.csv"}
KEY = ("dataset_split", "patient_id", "scan_id")
OUTPUT_COLUMNS = ["dataset_split", "patient_id", "scan_id", "ct_path", "mask_path",
                  "centroid_path", "coverage_type", "target_count", "validation_status"]
EXCLUDED_COLUMNS = OUTPUT_COLUMNS + ["exclusion_reason"]


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
        return list(reader)


def make_index(rows: list[dict[str, str]], label: str) -> dict[tuple[str, ...], dict[str, str]]:
    result = {}
    for row in rows:
        key = tuple((row.get(k) or "").strip() for k in KEY)
        if not all(key):
            raise ValueError(f"Blank key in {label}: {key}")
        if key in result:
            raise ValueError(f"Duplicate case in {label}: {key}")
        result[key] = row
    return result


def target_count(value: str, key: tuple[str, ...]) -> int:
    try:
        num = int(value)
        if num < 0 or num > 8:
            raise ValueError
        return num
    except (ValueError, TypeError):
        raise ValueError(f"Invalid target_count for {key}: {value!r}") from None


def build(inventory: list[dict[str, str]], report: list[dict[str, str]]):
    inv = make_index(inventory, "inventory")
    val = make_index(report, "validation report")
    if inv.keys() != val.keys():
        raise ValueError(f"Inventory/report cases differ: inventory-only={len(inv.keys()-val.keys())}, "
                         f"report-only={len(val.keys()-inv.keys())}")
    counts = Counter(k[0] for k in inv)
    if dict(counts) != EXPECTED:
        raise ValueError(f"Unexpected official split counts: {dict(counts)}; expected {EXPECTED}")
    patient_splits = {}
    for split, patient, _ in inv:
        patient_splits.setdefault(patient, set()).add(split)
    overlap = {p: s for p, s in patient_splits.items() if len(s) > 1}
    if overlap:
        raise ValueError(f"Patient leakage across splits: {overlap}")

    included = {s: [] for s in EXPECTED}
    excluded = []
    for key in sorted(inv):
        i, v = inv[key], val[key]
        split = key[0]
        ic = target_count((i.get("target_count") or "").strip(), key)
        vc = target_count((v.get("target_count") or "").strip(), key)
        if ic != vc:
            raise ValueError(f"target_count disagreement for {key}: inventory={ic}, validation={vc}")
        for field in ("ct_path", "mask_path"):
            if not (i.get(field) or "").strip() or not (v.get(field) or "").strip():
                raise ValueError(f"Missing {field} for {key}")
            if i[field].strip().replace('\\', '/') != v[field].strip().replace('\\', '/'):
                raise ValueError(f"{field} disagreement for {key}")
        status = (v.get("validation_status") or "").strip().lower()
        issues = (v.get("issues") or "").strip()
        if status not in {"passed", "warning", "failed"}:
            raise ValueError(f"Unexpected validation status for {key}: {status!r}")
        if status == "passed" and ic == 0:
            raise ValueError(f"Passed case has no target vertebrae: {key}")
        if status == "warning" and ic > 0:
            raise ValueError(f"Warning case has targets; review manually: {key}: {issues}")
        if status == "warning" and issues != "no_target_vertebrae":
            raise ValueError(f"Unexpected warning for {key}: {issues!r}")
        if status == "failed":
            raise ValueError(f"Failed validation case requires review before splitting: {key}: {issues}")
        row = {field: (i.get(field) or "").strip() for field in OUTPUT_COLUMNS}
        row["target_count"] = str(ic)
        row["validation_status"] = status
        if ic == 0:
            excluded.append({**row, "exclusion_reason": "no_target_vertebrae"})
        else:
            included[split].append(row)

    summary = {
        "source": "Official VerSe2020 splits (not randomly reshuffled)",
        "total_cases": len(inv),
        "original_split_counts": dict(EXPECTED),
        "eligible_split_counts": {s: len(included[s]) for s in EXPECTED},
        "excluded_split_counts": dict(Counter(r["dataset_split"] for r in excluded)),
        "excluded_total": len(excluded),
        "exclusion_reasons": dict(Counter(r["exclusion_reason"] for r in excluded)),
        "patient_overlap": 0,
        "note": "Eligible means technically validated and contains >=1 target vertebra (T10-L5); not clinical validation.",
    }
    return included, excluded, summary


def atomic_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig",
                                     dir=path.parent, prefix=".split_", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--inventory", type=Path, default=root / "outputs/m1_inspection/dataset_inventory.csv")
    parser.add_argument("--validation", type=Path, default=root / "outputs/m1_validation/validation_report.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs/m1_splits")
    args = parser.parse_args()
    inv = read_csv(args.inventory, set(KEY) | {"target_count", "ct_path", "mask_path"})
    val = read_csv(args.validation, set(KEY) | {"target_count", "ct_path", "mask_path", "validation_status", "issues"})
    included, excluded, summary = build(inv, val)  # All validation before writing any output.
    args.output.mkdir(parents=True, exist_ok=True)
    for split, filename in NAMES.items():
        atomic_csv(args.output / filename, OUTPUT_COLUMNS, included[split])
    atomic_csv(args.output / "excluded.csv", EXCLUDED_COLUMNS, excluded)
    dest = args.output / "split_summary.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output,
                                     prefix=".split_", suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, dest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
