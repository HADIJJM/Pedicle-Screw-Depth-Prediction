"""Batch inference: predict every case of a split with the model loaded once.

scripts/run_inference.py handles one case per call. This loop reuses the same inference
functions unchanged (checkpoint loading, fingerprint check, sliding window,
save_mask_two_spaces, verify_alignment) and adds what evaluation needs:
one consistent file name per case and a provenance file with the checkpoint hash.

  outputs/predictions/<run>/<split>/
    <scan_id>_pred.nii.gz           original CT grid, classes 0..8
    processed/<scan_id>_pred_2mm.nii.gz   (only with --save-processed)
    predictions_manifest.json

Examples (from the repository root):
  python scripts/predict_split.py --split validation \
      --checkpoint PATH/TO/best.pt --device cuda
  python scripts/predict_split.py --split test --final-test --checkpoint PATH --device cuda

Resumable: existing predictions made with the same checkpoint are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.module1_segmentation.data.dataset import load_manifest  # noqa: E402
from src.module1_segmentation.data.preprocess import portable_path, resolve_path  # noqa: E402
from src.module1_segmentation.inference.checkpoint import (  # noqa: E402
    build_model_from_checkpoint, load_checkpoint, verify_fingerprint,
)
from src.module1_segmentation.inference.save_output import (  # noqa: E402
    save_mask_two_spaces, verify_alignment,
)
from src.module1_segmentation.inference.sliding_window import sliding_window_inference  # noqa: E402

M2_MANIFESTS = {"training": "outputs/m2_preprocessing/train.csv",
                "validation": "outputs/m2_preprocessing/val.csv",
                "test": "outputs/m2_preprocessing/test.csv"}
PRED_PATTERN = "{scan_id}_pred.nii.gz"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (no automatic CPU fallback)")
    return device


def _write_manifest(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validation", choices=list(M2_MANIFESTS))
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--manifest", help="override the preprocessed manifest (default outputs/m2_preprocessing/<split>.csv)")
    p.add_argument("--output-dir", type=Path, help="default outputs/predictions/<run>/<split>")
    p.add_argument("--run-name", help="default: folder name of the checkpoint (e.g. baseline_96)")
    p.add_argument("--overlap", type=float, default=0.5, help="same default as scripts/run_inference.py")
    p.add_argument("--device", default="auto")
    p.add_argument("--save-processed", action="store_true", help="also keep the 2 mm prediction")
    p.add_argument("--force", action="store_true", help="recompute existing predictions")
    p.add_argument("--final-test", action="store_true", help="required for --split test")
    p.add_argument("--max-cases", type=int, default=0, help="debug only")
    a = p.parse_args()

    if a.split == "test" and not a.final_test:
        p.error("the test split is only for the final evaluation of the frozen model: add --final-test")
    ckpt_path = a.checkpoint.resolve()
    run_name = a.run_name or ckpt_path.parent.name or "run"
    out_dir = a.output_dir or ROOT / "outputs/predictions" / run_name / a.split
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device(a.device)
    t_load = time.time()
    ck = load_checkpoint(ckpt_path, device)
    model = build_model_from_checkpoint(ck, device)
    ck_sha = _sha256(ckpt_path)
    print(f"[info] {run_name}: epoch {ck['epoch']}, val Dice {ck['val_mean_dice']:.4f}, "
          f"sha256 {ck_sha[:12]}, device {device} ({time.time() - t_load:.1f}s)")

    rows = load_manifest(resolve_path(a.manifest or M2_MANIFESTS[a.split]))
    if {r["dataset_split"] for r in rows} != {a.split}:
        raise ValueError(f"Manifest does not contain only '{a.split}' cases")
    if {r["config_fingerprint"] for r in rows} != {ck["preprocess_fingerprint"]}:
        raise ValueError("Processed cases and checkpoint use different preprocessing fingerprints")
    if a.max_cases:
        rows = rows[:a.max_cases]

    man_path = out_dir / "predictions_manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    if manifest and not a.force:
        if manifest.get("checkpoint_sha256") != ck_sha:
            raise ValueError(f"{out_dir} holds predictions from another checkpoint; use a new --output-dir "
                             "or --force to replace them all")
        if manifest.get("overlap") != a.overlap:
            raise ValueError(f"{out_dir} was predicted with --overlap {manifest.get('overlap')}; "
                             "use the same value or --force")
    if a.force:
        manifest = {}
    manifest.update({
        "run_name": run_name, "dataset_split": a.split, "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ck_sha, "epoch": ck["epoch"], "val_mean_dice": ck["val_mean_dice"],
        "preprocess_fingerprint": ck["preprocess_fingerprint"], "patch_size_zyx": ck["patch_size_zyx"],
        "overlap": a.overlap, "device": str(device), "pred_pattern": PRED_PATTERN,
        "created_utc": manifest.get("created_utc", _now()), "updated_utc": _now(),
        "note": "inference functions (unchanged), batched by scripts/predict_split.py",
    })
    cases = manifest.setdefault("cases", {})

    done = failed = 0
    for k, row in enumerate(rows, 1):
        scan = row["scan_id"]
        out = out_dir / PRED_PATTERN.format(scan_id=scan)
        if out.is_file() and scan in cases and not a.force:
            print(f"[{k}/{len(rows)}] {scan}: exists, skipped")
            continue
        t0 = time.time()
        try:
            volume = np.load(resolve_path(row["image_path"]), allow_pickle=False).astype(np.float32, copy=False)
            meta = json.loads(resolve_path(row["metadata_path"]).read_text(encoding="utf-8"))
            verify_fingerprint(ck["preprocess_fingerprint"], meta)
            mask = sliding_window_inference(volume, model, tuple(ck["patch_size_zyx"]), device, overlap=a.overlap)
            processed = out_dir / "processed" / f"{scan}_pred_2mm.nii.gz" if a.save_processed else None
            save_mask_two_spaces(mask, meta, output_original=out, output_processed=processed)
            aligned = None
            ct = resolve_path(meta["source_ct"]) if meta.get("source_ct") else None
            if ct is not None and ct.is_file():
                verify_alignment(out, ct)
                aligned = True
            cases[scan] = {"patient_id": row["patient_id"], "file": out.name, "aligned_with_ct": aligned,
                           "classes": sorted(int(c) for c in np.unique(mask)),
                           "seconds": round(time.time() - t0, 1), "utc": _now()}
            done += 1
            print(f"[{k}/{len(rows)}] {scan}: classes {cases[scan]['classes']} ({cases[scan]['seconds']}s)")
        except Exception as exc:  # keep going; the case stays missing and the evaluator reports it
            failed += 1
            cases.pop(scan, None)
            if out.is_file():
                out.unlink()
            print(f"[{k}/{len(rows)}] {scan}: FAILED {type(exc).__name__}: {exc}")
        finally:
            _write_manifest(man_path, manifest)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"output_dir": portable_path(out_dir), "predicted": done, "failed": failed,
                      "total_in_folder": len(cases)}, indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
