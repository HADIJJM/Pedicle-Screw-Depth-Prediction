"""Inspect VerSe2020 CT scans and masks with low RAM and resumable checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = ROOT / 'data' / 'raw' / 'VerSe2020'
DEFAULT_OUTPUT_DIR = ROOT / 'outputs' / 'm1_inspection'
TARGETS = {17:'T10',18:'T11',19:'T12',20:'L1',21:'L2',22:'L3',23:'L4',24:'L5'}
SPLIT_NAMES = {'dataset-01training':'training','dataset-02validation':'validation','dataset-03test':'test'}
COLUMNS = ['dataset_split','patient_id','scan_id','ct_path','mask_path','centroid_path','mask_exists','centroid_exists','ct_shape','mask_shape','spacing_mm','orientation','affine','all_labels','target_vertebrae','target_count','coverage_type','inspection_status','error']


def relpath(path: Path | None) -> str:
    if path is None:
        return ''
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_mask_labels(mask_path: Path, slab: int) -> set[int]:
    """Read narrow slabs from compressed NIfTI without loading entire volume.

    Use unscaled values when the mask header has identity scaling. The slice proxy
    may still incur gzip seek overhead, but peak array memory stays small.
    """
    img = nib.load(str(mask_path), mmap=False, keep_file_open=False)
    if len(img.shape) != 3:
        raise ValueError(f'Expected 3D mask; found shape {img.shape}')
    labels = set()
    for z in range(0, img.shape[2], slab):
        stop = min(z + slab, img.shape[2])
        # Get one small slab. Avoid np.unique on large temporary float64 arrays.
        block = np.asarray(img.dataobj[:, :, z:stop])
        if not np.isfinite(block).all():
            raise ValueError('Non-finite mask label')
        # Values in vertebral masks are small integers; inspect one slab at a time.
        vals = np.unique(block)
        for v in vals:
            f = float(v)
            if not f.is_integer():
                raise ValueError(f'Non-integer mask label: {f}')
            labels.add(int(f))
        del block, vals
    del img
    gc.collect()
    return labels


def scan_one(ct_path: Path, derivatives: Path, split: str, slab: int) -> dict:
    patient_id = ct_path.parent.name
    scan_id = ct_path.name.removesuffix('_ct.nii.gz')
    mask_candidate = derivatives / patient_id / f'{scan_id}_seg-vert_msk.nii.gz'
    centroid_candidate = derivatives / patient_id / f'{scan_id}_seg-subreg_ctd.json'
    mask = mask_candidate if mask_candidate.is_file() else None
    centroid = centroid_candidate if centroid_candidate.is_file() else None
    rec = dict.fromkeys(COLUMNS, '')
    rec.update(dataset_split=split, patient_id=patient_id, scan_id=scan_id,
               ct_path=relpath(ct_path), mask_path=relpath(mask),
               centroid_path=relpath(centroid), mask_exists=mask is not None,
               centroid_exists=centroid is not None, target_count=0,
               coverage_type='unknown', inspection_status='pending')
    try:
        ct_img = nib.load(str(ct_path), mmap=False)
        rec['ct_shape'] = json.dumps(list(ct_img.shape))
        rec['spacing_mm'] = json.dumps([round(float(v), 5) for v in ct_img.header.get_zooms()[:3]])
        rec['orientation'] = ''.join(nib.aff2axcodes(ct_img.affine))
        rec['affine'] = json.dumps(np.asarray(ct_img.affine).round(6).tolist())
        del ct_img
        if mask is None:
            rec['inspection_status'] = 'missing_mask'
        else:
            mask_img = nib.load(str(mask), mmap=False)
            rec['mask_shape'] = json.dumps(list(mask_img.shape))
            del mask_img
            labels = read_mask_labels(mask, slab)
            rec['all_labels'] = ','.join(map(str, sorted(labels)))
            names = [name for label, name in TARGETS.items() if label in labels]
            rec['target_vertebrae'] = ','.join(names)
            rec['target_count'] = len(names)
            rec['coverage_type'] = 'full' if len(names) == len(TARGETS) else 'partial' if names else 'none'
            rec['inspection_status'] = 'inspected'
    except (Exception, MemoryError) as exc:
        rec['inspection_status'] = 'error'
        rec['error'] = f'{type(exc).__name__}: {exc}'
    gc.collect()
    return rec


def save_inventory(records: dict[tuple[str, str], dict], path: Path) -> None:
    df = pd.DataFrame.from_records(list(records.values()), columns=COLUMNS)
    temp = path.with_suffix('.csv.tmp')
    df.to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, path)


def write_summary(records: dict[tuple[str, str], dict], path: Path) -> dict:
    df = pd.DataFrame.from_records(list(records.values()), columns=COLUMNS)
    counts = df.groupby('patient_id').size()
    overlap = df.groupby('patient_id')['dataset_split'].nunique()
    summary = {
        'scans': int(len(df)),
        'unique_patient_ids': int(df['patient_id'].nunique()),
        'scans_per_split': {str(k): int(v) for k,v in Counter(df['dataset_split']).items()},
        'inspection_status': {str(k): int(v) for k,v in Counter(df['inspection_status']).items()},
        'coverage_type': {str(k): int(v) for k,v in Counter(df['coverage_type']).items()},
        'patients_with_multiple_scans': {str(k): int(v) for k,v in counts.items() if v > 1},
        'patient_ids_across_splits': [str(x) for x in overlap[overlap > 1].index],
        'note': 'Inventory only; spatial alignment and clinical validity are NOT validated.'
    }
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--split', choices=['all','training','validation','test'], default='all')
    parser.add_argument('--block-size', type=int, default=1, help='Number of slices per read (default 1 for 8GB RAM).')
    parser.add_argument('--fresh', action='store_true', help='Ignore cached results and rescan selected cases.')
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error('--block-size must be >= 1')
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        parser.error(f'Dataset directory not found: {dataset_root}')
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = output_dir / 'dataset_inventory.csv'
    summary_path = output_dir / 'inspection_summary.json'
    records: dict[tuple[str,str], dict] = {}
    if inventory.exists():
        prior = pd.read_csv(inventory, dtype=str, keep_default_na=False)
        if all(c in prior.columns for c in COLUMNS):
            for row in prior.to_dict('records'):
                records[(row['dataset_split'], row['scan_id'])] = row
    discovered = []
    for folder_name, split in SPLIT_NAMES.items():
        if args.split not in ('all', split):
            continue
        folder = dataset_root / folder_name
        raw = folder / 'rawdata'
        if raw.is_dir():
            for ct_path in sorted(raw.rglob('*_ct.nii.gz')):
                discovered.append((ct_path, folder / 'derivatives', split))
    if not discovered:
        parser.error('No CT files found for selected split')
    print(f'Discovered {len(discovered)} scans for {args.split}; cached records: {len(records)}', flush=True)
    for index, (ct_path, derivatives, split) in enumerate(discovered, 1):
        scan_id = ct_path.name.removesuffix('_ct.nii.gz')
        key = (split, scan_id)
        old = records.get(key)
        if old and old.get('inspection_status') in ('inspected','missing_mask') and not args.fresh:
            rec = old
            action = 'reused'
        else:
            rec = scan_one(ct_path, derivatives, split, args.block_size)
            records[key] = rec
            save_inventory(records, inventory)
            write_summary(records, summary_path)
            action = 'checked'
        print(f'[{index}/{len(discovered)}] {split}/{scan_id}: '
              f"{rec['inspection_status']} | {rec['coverage_type']} | {action}", flush=True)
    summary = write_summary(records, summary_path)
    print('\n=== INSPECTION SUMMARY ===')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nInventory: {inventory}\nSummary: {summary_path}')

if __name__ == '__main__':
    main()
