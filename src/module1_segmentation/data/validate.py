"""Memory-bounded, resumable VerSe2020 dataset validation.

Each case runs in a separate subprocess. Gzipped masks are decompressed to
temporary disk files, then inspected slice-by-slice. Existing passed/warning
rows are reused. Run from any working directory.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INVENTORY = (
    ROOT / "outputs" / "m1_inspection" / "dataset_inventory.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "m1_validation"

VALID_LABELS = set(range(29))  # 0 through 28
TARGET_LABELS = set(range(17, 25))  # T10 through L5

AFFINE_ATOL = 1e-3
SPACING_ATOL = 1e-3

REPORT_COLUMNS = [
    "dataset_split",
    "patient_id",
    "scan_id",
    "ct_path",
    "mask_path",
    "centroid_path",
    "coverage_type",
    "ct_shape",
    "mask_shape",
    "ct_spacing_mm",
    "mask_spacing_mm",
    "ct_orientation",
    "mask_orientation",
    "shape_match",
    "spacing_match",
    "affine_match",
    "orientation_match",
    "mask_integer_labels",
    "mask_finite_labels",
    "unexpected_labels",
    "target_count",
    "centroid_status",
    "validation_status",
    "issues",
]


# ============================================================
# 2. Helper functions
# ============================================================

def resolve_path(value):
    """Convert inventory paths into absolute paths."""

    if pd.isna(value) or not str(value).strip():
        return None

    path = Path(str(value))

    if not path.is_absolute():
        path = ROOT / path

    return path


def json_list(values):
    return json.dumps(
        [float(x) for x in values]
    )


def check_centroid(path):
    """
    Check whether a centroid JSON file can be parsed.

    Detailed coordinate validation is handled separately.
    """

    if path is None:
        return "missing"

    if not path.is_file():
        return "missing"

    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return "invalid_structure"

        if len(data) == 0:
            return "empty"

        return "readable"

    except (OSError, ValueError, UnicodeError):
        return "invalid_json"


# ============================================================
# 3. Validate one case
# ============================================================

def validate_case(row):

    ct_path = resolve_path(row.get("ct_path"))
    mask_path = resolve_path(row.get("mask_path"))
    centroid_path = resolve_path(row.get("centroid_path"))

    record = {
        "dataset_split": row.get("dataset_split", ""),
        "patient_id": row.get("patient_id", ""),
        "scan_id": row.get("scan_id", ""),
        "ct_path": row.get("ct_path", ""),
        "mask_path": row.get("mask_path", ""),
        "centroid_path": row.get("centroid_path", ""),
        "coverage_type": row.get("coverage_type", ""),
        "ct_shape": "",
        "mask_shape": "",
        "ct_spacing_mm": "",
        "mask_spacing_mm": "",
        "ct_orientation": "",
        "mask_orientation": "",
        "shape_match": False,
        "spacing_match": False,
        "affine_match": False,
        "orientation_match": False,
        "mask_integer_labels": False,
        "mask_finite_labels": False,
        "unexpected_labels": "",
        "target_count": 0,
        "centroid_status": "",
        "validation_status": "pending",
        "issues": "",
    }

    issues = []
    warnings = []

    # --------------------------------------------------------
    # File existence
    # --------------------------------------------------------

    if ct_path is None or not ct_path.is_file():
        issues.append("missing_ct")

    if mask_path is None or not mask_path.is_file():
        issues.append("missing_mask")

    if issues:
        record["validation_status"] = "failed"
        record["issues"] = ";".join(issues)
        return record

    try:
        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))

        record["ct_shape"] = str(ct_img.shape)
        record["mask_shape"] = str(mask_img.shape)

        ct_spacing = np.asarray(
            ct_img.header.get_zooms()[:3],
            dtype=float,
        )

        mask_spacing = np.asarray(
            mask_img.header.get_zooms()[:3],
            dtype=float,
        )

        record["ct_spacing_mm"] = json_list(ct_spacing)
        record["mask_spacing_mm"] = json_list(mask_spacing)

        record["ct_orientation"] = "".join(
            nib.aff2axcodes(ct_img.affine)
        )

        record["mask_orientation"] = "".join(
            nib.aff2axcodes(mask_img.affine)
        )

        # ----------------------------------------------------
        # Shape validation
        # ----------------------------------------------------

        if len(ct_img.shape) != 3:
            issues.append("ct_not_3d")

        if len(mask_img.shape) != 3:
            issues.append("mask_not_3d")

        record["shape_match"] = (
            ct_img.shape == mask_img.shape
        )

        if not record["shape_match"]:
            issues.append("shape_mismatch")

        # ----------------------------------------------------
        # Spacing validation
        # ----------------------------------------------------

        record["spacing_match"] = bool(
            np.allclose(
                ct_spacing,
                mask_spacing,
                rtol=0,
                atol=SPACING_ATOL,
            )
        )

        if not record["spacing_match"]:
            issues.append("spacing_mismatch")

        if (
            not np.isfinite(ct_spacing).all()
            or np.any(ct_spacing <= 0)
        ):
            issues.append("invalid_ct_spacing")

        if (
            not np.isfinite(mask_spacing).all()
            or np.any(mask_spacing <= 0)
        ):
            issues.append("invalid_mask_spacing")

        # ----------------------------------------------------
        # Affine validation
        # ----------------------------------------------------

        record["affine_match"] = bool(
            np.allclose(
                ct_img.affine,
                mask_img.affine,
                rtol=0,
                atol=AFFINE_ATOL,
            )
        )

        if not record["affine_match"]:
            issues.append("affine_mismatch")

        record["orientation_match"] = (
            record["ct_orientation"]
            == record["mask_orientation"]
        )

        if not record["orientation_match"]:
            issues.append("orientation_mismatch")

        if (
            not np.isfinite(ct_img.affine).all()
            or not np.isfinite(mask_img.affine).all()
        ):
            issues.append("invalid_affine")

        # ----------------------------------------------------
        # Mask label validation
        # ----------------------------------------------------

        # Process one slice at a time to keep RAM usage bounded.
        labels = set()
        finite = True
        integer_labels = True
        if len(mask_img.shape) == 3:
            for z in range(mask_img.shape[2]):
                slab = np.asarray(mask_img.dataobj[:, :, z:z + 1])
                slab_finite = bool(np.isfinite(slab).all())
                if not slab_finite:
                    finite = False
                else:
                    slab_integer = bool(np.equal(slab, np.rint(slab)).all())
                    if not slab_integer:
                        integer_labels = False
                    else:
                        labels.update(int(x) for x in np.unique(slab))
                del slab
        else:
            finite = False
            integer_labels = False

        record["mask_finite_labels"] = finite
        record["mask_integer_labels"] = finite and integer_labels
        if not finite:
            issues.append("non_finite_mask_labels")
        if finite and not integer_labels:
            issues.append("non_integer_mask_labels")
        if finite and integer_labels:
            unexpected = sorted(labels - VALID_LABELS)
            record["unexpected_labels"] = ",".join(map(str, unexpected))
            if unexpected:
                issues.append("unexpected_mask_labels")
            record["target_count"] = len(labels & TARGET_LABELS)
            if record["target_count"] == 0:
                warnings.append("no_target_vertebrae")
            previous_count = row.get("target_count")
            if pd.notna(previous_count) and str(previous_count).strip():
                if int(float(previous_count)) != record["target_count"]:
                    issues.append("target_count_changed")

        # ----------------------------------------------------
        # Centroid JSON
        # ----------------------------------------------------

        record["centroid_status"] = check_centroid(
            centroid_path
        )

        if record["centroid_status"] != "readable":
            warnings.append(
                f"centroid_{record['centroid_status']}"
            )

        # ----------------------------------------------------
        # Final status
        # ----------------------------------------------------

        if issues:
            record["validation_status"] = "failed"

        elif warnings:
            record["validation_status"] = "warning"

        else:
            record["validation_status"] = "passed"

        record["issues"] = ";".join(
            issues + warnings
        )

    except Exception as exc:
        record["validation_status"] = "failed"

        issues.append(
            f"{type(exc).__name__}: {exc}"
        )

        record["issues"] = ";".join(issues)

    return record


# ============================================================
# 4. Main
# ============================================================


import argparse
import csv
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPORT = OUTPUT_DIR / 'validation_report.csv'
SUMMARY = OUTPUT_DIR / 'validation_summary.json'

def read_csv(path):
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def atomic_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(.5)
    rescue = path.with_name(path.stem + '_RESCUE.csv')
    shutil.copy2(tmp, rescue)
    raise PermissionError(f'Close the CSV in Excel. Progress saved to {rescue}')


def key(row):
    return (str(row.get('dataset_split','')), str(row.get('scan_id','')), str(row.get('patient_id','')))


def worker(input_path, output_path):
    row = json.loads(Path(input_path).read_text(encoding='utf-8'))
    mask_path = resolve_path(row.get('mask_path'))
    original_load = nib.load
    with tempfile.TemporaryDirectory(prefix='verse_validation_') as temp:
        if mask_path is not None and mask_path.is_file() and str(mask_path).lower().endswith('.gz'):
            raw_path = Path(temp) / 'mask.nii'
            with gzip.open(mask_path, 'rb') as source, raw_path.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            def load_image(filename, *args, **kwargs):
                if Path(filename).resolve() == mask_path.resolve():
                    return original_load(str(raw_path), mmap=True)
                return original_load(filename, *args, **kwargs)
            nib.load = load_image
        try:
            result = validate_case(row)
        finally:
            nib.load = original_load
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')


def main(limit, timeout):
    inventory = read_csv(INVENTORY)
    if not inventory:
        raise RuntimeError(f'Missing/empty inventory: {INVENTORY}')
    previous = read_csv(REPORT)
    if previous:
        backup = OUTPUT_DIR / 'validation_report_before_resume.csv'
        if not backup.exists():
            shutil.copy2(REPORT, backup)
            print(f'Backup: {backup}', flush=True)
    columns = REPORT_COLUMNS
    inventory_keys = [key(r) for r in inventory]
    if len(set(inventory_keys)) != len(inventory_keys):
        raise RuntimeError('Duplicate scan keys in inventory; stopping to avoid mixing results.')
    known = {key(r): r for r in previous if key(r) in set(inventory_keys)}
    pending = [r for r in inventory if known.get(key(r), {}).get('validation_status') not in ('passed', 'warning')]
    print(f'Inventory: {len(inventory)} | saved: {len(known)} | to check/retry: {len(pending)}', flush=True)
    if limit:
        pending = pending[:limit]
    for index, row in enumerate(pending, 1):
        label = f"{row.get('dataset_split','')}/{row.get('scan_id',row.get('patient_id',''))}"
        print(f'[{index}/{len(pending)}] {label}: checking...', flush=True)
        with tempfile.TemporaryDirectory(prefix='verse_worker_io_') as temp:
            inp, out = Path(temp) / 'in.json', Path(temp) / 'out.json'
            inp.write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
            try:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(inp), str(out)], cwd=str(ROOT), timeout=timeout, capture_output=True, text=True)
                if process.returncode != 0 or not out.exists():
                    raise RuntimeError((process.stderr or process.stdout or f'worker exit {process.returncode}')[-400:])
                result = json.loads(out.read_text(encoding='utf-8'))
            except (subprocess.TimeoutExpired, Exception) as exc:
                result = {c: '' for c in columns}
                result.update({c: row.get(c, '') for c in ('dataset_split', 'patient_id', 'scan_id', 'ct_path', 'mask_path', 'centroid_path', 'coverage_type')})
                result['validation_status'] = 'failed'
                result['issues'] = f'worker_error: {type(exc).__name__}: {str(exc)[:300]}'
        known[key(row)] = result
        ordered = [known[key(r)] for r in inventory if key(r) in known]
        atomic_csv(REPORT, ordered, columns)
        print(f"  -> {result['validation_status']} | {result.get('issues') or 'OK'}", flush=True)
    ordered = [known[key(r)] for r in inventory if key(r) in known]
    from collections import Counter
    counts = Counter(r.get('validation_status', '') for r in ordered)
    summary = {'total_scans': len(ordered), 'expected_scans': len(inventory), 'passed': counts['passed'], 'warnings': counts['warning'], 'failed': counts['failed'], 'remaining': len(inventory)-len(ordered), 'status_counts': dict(counts), 'note': 'Technical dataset validation only; passing does not imply clinical safety.'}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSUMMARY:', json.dumps(summary, indent=2), flush=True)
    print(f'Report: {REPORT}\nSummary: {SUMMARY}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=2, metavar=('INPUT', 'OUTPUT'))
    parser.add_argument('--limit', type=int, default=0, help='Only process this many pending cases; 0 means all')
    parser.add_argument('--timeout', type=int, default=900, help='Seconds per scan')
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        main(args.limit, args.timeout)
