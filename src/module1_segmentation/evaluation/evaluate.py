"""Evaluate predicted vertebra masks against the VerSe2020 reference masks.

Reads the split manifest (original CT + reference mask paths), matches
each case to its original-space prediction by scan_id, scores it on the
reference grid and writes tables, figures and a short report:

  outputs/evaluation/<tag>/<split>/
    per_case.csv  per_level.csv  summary_per_level.csv  failure_cases.csv
    confusion_voxels.csv  level_assignment.csv  summary.json  report.md
    cases/<scan_id>.json (per-case cache, used to resume)
    figures/  (summary plots + one overlay per case)

The test split is evaluated only with --final-test and is locked to the first
checkpoint used (outputs/evaluation/test_lock.json). Research prototype.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.module1_segmentation.data.preprocess import PROJECT_ROOT, portable_path, resolve_path
from src.module1_segmentation.evaluation import metrics as M
from src.module1_segmentation.evaluation.case_io import load_case, load_ct_for_display

METRICS_VERSION = "eval-v1"
SPLIT_MANIFESTS = {
    "training": "outputs/m1_splits/train.csv",
    "validation": "outputs/m1_splits/val.csv",
    "test": "outputs/m1_splits/test.csv",
}
MANIFEST_REQUIRED = {"dataset_split", "patient_id", "scan_id", "ct_path", "mask_path"}
DEFAULT_PRED_PATTERN = "{scan_id}_pred.nii.gz"
PRED_MANIFEST_NAME = "predictions_manifest.json"
TEST_LOCK = "outputs/evaluation/test_lock.json"

LEVEL_COLUMNS = ["dataset_split", "patient_id", "scan_id", "level", "class", "status", "outcome",
                 "dice", "iou", "gt_voxels", "pred_voxels", "tp_voxels", "gt_volume_ml", "pred_volume_ml",
                 "assigned_label", "assigned_fraction", "pred_sits_on", "centroid_distance_mm",
                 "identified", "pred_components", "largest_component_fraction"]
CASE_COLUMNS = (["dataset_split", "patient_id", "scan_id", "failure_category", "mean_dice", "mean_iou",
                 "spine_dice", "spine_iou", "best_shift", "mean_dice_at_best_shift", "shift_gain",
                 "n_levels_gt", "n_levels_pred", "levels_gt", "n_correct", "n_wrong_level", "n_missed",
                 "n_false_positive", "n_identified", "id_rate", "fragmented_levels"]
                + [f"dice_{n}" for n in M.LEVEL_NAMES]
                + ["grid_resampled", "grid_note", "train_val_mean_dice", "delta_vs_train_val", "figure", "prediction"])


@dataclass
class EvalConfig:
    pred_dir: str
    split: str = "validation"
    manifest: str | None = None          # default: outputs/m1_splits/<split>.csv
    pred_pattern: str = DEFAULT_PRED_PATTERN
    pred_labels: str = "model"           # model (0..8) | verse (17..24)
    output_root: str = "outputs/evaluation"
    tag: str | None = None               # default: run name from predictions_manifest.json
    checkpoint: str | None = None        # optional; hashed for provenance
    figures: str = "all"                 # all | failures | none
    reference_dice_csv: str | None = None  # e.g. training-run case_dice_best.csv (2 mm grid)
    allow_missing: bool = False
    strict_grid: bool = False
    resume: bool = True
    final_test: bool = False
    allow_new_test_checkpoint: bool = False
    max_cases: int = 0                   # DEBUG ONLY
    max_shift: int = 3


# ------------------------------------------------------------------ helpers
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _clean(v: Any) -> Any:
    """NaN -> None so JSON stays valid."""
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean(data), indent=2, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in columns})


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{float(v):.5f}"
    return v


def read_split_manifest(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:  # split CSVs carry a BOM
        reader = csv.DictReader(f)
        if not reader.fieldnames or not MANIFEST_REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(MANIFEST_REQUIRED)}")
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]
    if not rows:
        raise ValueError(f"No cases in {path}")
    ids = [r["scan_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scan_id in {path}")
    return rows


def load_prediction_provenance(pred_dir: Path) -> dict[str, Any]:
    path = pred_dir / PRED_MANIFEST_NAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_info(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    info: dict[str, Any] = {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path)}
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
        info.update(epoch=ck.get("epoch"), val_mean_dice=ck.get("val_mean_dice"),
                    preprocess_fingerprint=ck.get("preprocess_fingerprint"))
    except Exception as exc:  # torch missing or unreadable: the hash is what matters
        info["note"] = f"checkpoint metadata not read ({type(exc).__name__})"
    return info


def resolve_provenance(cfg: EvalConfig, pred_dir: Path) -> dict[str, Any]:
    prov = load_prediction_provenance(pred_dir)
    if prov and prov.get("dataset_split") and prov["dataset_split"] != cfg.split:
        raise ValueError(f"{pred_dir} holds predictions for split '{prov['dataset_split']}', not '{cfg.split}'")
    info = {k: prov.get(k) for k in ("run_name", "checkpoint", "checkpoint_sha256", "epoch",
                                     "val_mean_dice", "overlap", "created_utc") if k in prov}
    if cfg.checkpoint:
        ck = checkpoint_info(cfg.checkpoint)
        if info.get("checkpoint_sha256") and info["checkpoint_sha256"] != ck["checkpoint_sha256"]:
            raise ValueError("--checkpoint differs from the checkpoint recorded in predictions_manifest.json")
        info.update({k: v for k, v in ck.items() if v is not None})
    return info


def enforce_test_protocol(cfg: EvalConfig, provenance: dict[str, Any], project_root: Path) -> None:
    """Test = one final evaluation with a frozen model. Locks the checkpoint hash."""
    if cfg.split != "test":
        return
    if not cfg.final_test:
        raise PermissionError("The test split is reserved for the single final evaluation of the frozen "
                              "model. Develop on --split validation; pass --final-test only at the end.")
    sha = provenance.get("checkpoint_sha256")
    if not sha:
        raise ValueError("Final test evaluation needs checkpoint provenance: use predictions made by "
                         "scripts/predict_split.py (predictions_manifest.json) or pass --checkpoint.")
    if cfg.max_cases or cfg.allow_missing:
        raise ValueError("Final test evaluation must cover every test case (no --max-cases/--allow-missing).")
    lock_path = resolve_path(TEST_LOCK, project_root)
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["checkpoint_sha256"] != sha and not cfg.allow_new_test_checkpoint:
            raise PermissionError(
                f"Test split already evaluated with checkpoint {lock['checkpoint_sha256'][:12]} on "
                f"{lock['created_utc']}. Re-testing a different model turns the test set into a tuning "
                "set. Only override with --allow-new-test-checkpoint and report it.")
        if lock["checkpoint_sha256"] != sha:
            lock.setdefault("overrides", []).append({"checkpoint_sha256": sha, "utc": _now()})
            _write_json(lock_path, lock)
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(lock_path, {"created_utc": _now(), "checkpoint_sha256": sha,
                            "checkpoint": provenance.get("checkpoint"), "epoch": provenance.get("epoch"),
                            "tag": cfg.tag, "git_commit": _git_commit()})


def read_reference_dice(path: str | Path | None) -> dict[str, float]:
    """Optional training-run case_dice_best.csv -> {scan_id: mean Dice over levels (2 mm grid)}."""
    if not path:
        return {}
    out = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            vals = [float(r[f"dice_{n}"]) for n in M.LEVEL_NAMES if r.get(f"dice_{n}", "") != ""]
            out[r["scan_id"]] = float(np.mean(vals)) if vals else float("nan")
    return out


# --------------------------------------------------------------- per case
def _case_record(row: dict[str, str], result: dict[str, Any], vols, pred_path: Path,
                 pred_sha: str, project_root: Path) -> dict[str, Any]:
    case = dict(result["case"])
    case.update(dataset_split=row["dataset_split"], patient_id=row["patient_id"], scan_id=row["scan_id"],
                grid_resampled=int(vols.grid_resampled), grid_note=vols.grid_note,
                prediction=portable_path(pred_path, project_root))
    for lvl in result["levels"]:
        case[f"dice_{lvl['level']}"] = lvl["dice"]
    levels = [dict(lvl, dataset_split=row["dataset_split"], patient_id=row["patient_id"],
                   scan_id=row["scan_id"]) for lvl in result["levels"]]
    return {"metrics_version": METRICS_VERSION, "prediction_sha256": pred_sha,
            "reference_mask": row["mask_path"], "case": case, "levels": levels,
            "confusion": result["confusion"].tolist(),
            "mean_dice_by_shift": {str(k): v for k, v in result["mean_dice_by_shift"].items()},
            "voxel_size_mm": list(vols.zooms)}


def _load_cached(path: Path, pred_sha: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if rec.get("metrics_version") != METRICS_VERSION or rec.get("prediction_sha256") != pred_sha:
        return None
    return rec


def _restore_nan(d: dict[str, Any]) -> dict[str, Any]:
    return {k: (float("nan") if v is None and k in _FLOAT_KEYS else v) for k, v in d.items()}


_FLOAT_KEYS = {"dice", "iou", "assigned_fraction", "centroid_distance_mm", "largest_component_fraction",
               "mean_dice", "mean_iou", "spine_dice", "spine_iou", "mean_dice_at_best_shift", "shift_gain",
               "id_rate", *[f"dice_{n}" for n in M.LEVEL_NAMES]}


# ----------------------------------------------------------------- summary
def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if not arr.size:
        return {k: float("nan") for k in ("mean", "std", "median", "q1", "q3", "min", "max")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)), "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)), "min": float(arr.min()), "max": float(arr.max())}


def summarize(cases: list[dict[str, Any]], levels: list[dict[str, Any]], confusion: np.ndarray) -> dict[str, Any]:
    per_level = []
    for name in M.LEVEL_NAMES:
        rows = [r for r in levels if r["level"] == name]
        gt_rows = [r for r in rows if r["gt_voxels"] > 0]
        d, i = _stats([r["dice"] for r in rows]), _stats([r["iou"] for r in rows])
        outcomes = [r["outcome"] for r in rows]
        per_level.append({
            "level": name, "n_cases_gt": len(gt_rows),
            "n_cases_scored": sum(not np.isnan(r["dice"]) for r in rows),
            **{f"dice_{k}": v for k, v in d.items()}, **{f"iou_{k}": v for k, v in i.items()},
            "n_correct": outcomes.count("correct"), "n_wrong_level": outcomes.count("wrong_level"),
            "n_missed": outcomes.count("missed"), "n_false_positive": outcomes.count("false_positive"),
            "id_rate": (sum(r["identified"] == 1 for r in gt_rows) / len(gt_rows)) if gt_rows else float("nan"),
        })
    gt_total = sum(c["n_levels_gt"] for c in cases)
    level_means = [r["dice_mean"] for r in per_level]
    categories = {k: 0 for k in M.FAILURE_CATEGORIES}
    shifts: dict[str, int] = {}
    for c in cases:
        categories[c["failure_category"]] += 1
        shifts[str(c["best_shift"])] = shifts.get(str(c["best_shift"]), 0) + 1
    overall = {
        "mean_dice_levels": M.nanmean(level_means),  # training-time definition (mean of per-level means)
        "mean_iou_levels": M.nanmean([r["iou_mean"] for r in per_level]),
        "mean_dice_cases": M.nanmean([c["mean_dice"] for c in cases]),
        "median_dice_cases": _stats([c["mean_dice"] for c in cases])["median"],
        "mean_spine_dice": M.nanmean([c["spine_dice"] for c in cases]),
        "mean_dice_at_best_shift": M.nanmean([c["mean_dice_at_best_shift"] for c in cases]),
        "identification_rate": (sum(c["n_identified"] for c in cases) / gt_total) if gt_total else float("nan"),
    }
    return {"overall": overall, "per_level": per_level, "failure_categories": categories,
            "best_shift_counts": dict(sorted(shifts.items(), key=lambda kv: int(kv[0]))),
            "confusion_voxels": confusion.tolist()}


def level_assignment(levels: list[dict[str, Any]]) -> np.ndarray:
    """Cases counted by (reference level, label the prediction mostly gave it); last column = missed."""
    cols = list(M.LEVEL_NAMES) + ["background"]
    out = np.zeros((8, 9), dtype=np.int64)
    for r in levels:
        if r["gt_voxels"] > 0 and r["assigned_label"]:
            out[r["class"] - 1, cols.index(r["assigned_label"])] += 1
    return out


# ---------------------------------------------------------------- report
def write_report(path: Path, summary: dict[str, Any], cases: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    o = summary["overall"]

    def f(v, nd=3):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

    prov = meta.get("provenance", {})
    lines = [
        f"# Segmentation evaluation: {meta['tag']} / {meta['split']}", "",
        f"Generated {meta['created_utc']} (git {meta.get('git_commit') or 'n/a'}). "
        "Research prototype, not for clinical use.", "",
        f"- Cases evaluated: **{meta['n_cases']}** (missing predictions: {meta['n_missing']}, "
        f"prediction grid resampled: {meta['n_grid_resampled']})",
        f"- Checkpoint: {prov.get('checkpoint') or 'not recorded'}"
        + (f" (sha256 {prov['checkpoint_sha256'][:12]}, epoch {prov.get('epoch')})" if prov.get("checkpoint_sha256") else ""),
        "- Grid: original CT grid of every case (the validation Dice logged during training used the 2 mm grid).", "",
        "## Headline", "",
        "| Metric | Value |", "|---|---:|",
        f"| Mean Dice over levels (same definition as training) | {f(o['mean_dice_levels'])} |",
        f"| Mean IoU over levels | {f(o['mean_iou_levels'])} |",
        f"| Mean of case Dice | {f(o['mean_dice_cases'])} |",
        f"| Median of case Dice | {f(o['median_dice_cases'])} |",
        f"| Spine Dice (all target vertebrae as one class) | {f(o['mean_spine_dice'])} |",
        f"| Mean Dice if each case's level names were shifted optimally | {f(o['mean_dice_at_best_shift'])} |",
        f"| Identification rate (VerSe style, 20 mm) | {f(o['identification_rate'])} |", "",
        "A spine Dice well above the per-level Dice means the bone is found but levels are misnamed.", "",
        "## Per level", "",
        "| Level | Cases | Dice mean ± SD | Dice median | IoU mean | Correct | Wrong level | Missed | False pos. | ID rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["per_level"]:
        lines.append(f"| {r['level']} | {r['n_cases_gt']} | {f(r['dice_mean'])} ± {f(r['dice_std'])} | "
                     f"{f(r['dice_median'])} | {f(r['iou_mean'])} | {r['n_correct']} | {r['n_wrong_level']} | "
                     f"{r['n_missed']} | {r['n_false_positive']} | {f(r['id_rate'])} |")
    lines += ["", "## Failure categories", "", "| Category | Cases |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in summary["failure_categories"].items()]
    lines += ["", "## Best uniform level shift per case", "",
              "Shift +1 = the model named each vertebra one level too cranially (true L3 called L2).", "",
              "| Shift | Cases |", "|---:|---:|"]
    lines += [f"| {int(k):+d} | {v} |" for k, v in summary["best_shift_counts"].items()]
    worst = sorted(cases, key=lambda c: (np.nan_to_num(c["mean_dice"], nan=-1.0)))[:10]
    lines += ["", "## Ten lowest cases", "",
              "| Case | Category | Dice | Spine Dice | Best shift → Dice | Missed | Wrong level | Figure |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for c in worst:
        fig = f"[png]({c['figure']})" if c.get("figure") else ""
        lines.append(f"| {c['scan_id']} | {c['failure_category']} | {f(c['mean_dice'])} | {f(c['spine_dice'])} | "
                     f"{int(c['best_shift']):+d} → {f(c['mean_dice_at_best_shift'])} | {c['n_missed']} | "
                     f"{c['n_wrong_level']} | {fig} |")
    lines += ["", "## Definitions", "",
              "- Dice/IoU per level on the reference grid; a level absent in both masks is excluded, "
              "a level present in only one mask scores 0.",
              f"- Outcome per reference level: *missed* if < {M.MISSED_COVERAGE:.0%} of its voxels are "
              "covered by any predicted vertebra; otherwise *correct* or *wrong_level* by the label covering most of it.",
              f"- Failure category, first rule that matches: no_prediction; poor_segmentation (spine Dice < {M.POOR_SPINE_DICE}); "
              f"level_shift (a non-zero shift raises mean Dice by ≥ {M.SHIFT_GAIN}); wrong_level_labels; missed_levels; "
              f"false_positive_levels; low_overlap (mean Dice < {M.OK_MEAN_DICE}); ok.",
              f"- Identification: predicted centroid within {M.ID_DISTANCE_MM:.0f} mm of the reference centroid "
              "of the same level, and that centroid is the nearest reference centroid.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------- main
def evaluate_split(cfg: EvalConfig, *, project_root: Path = PROJECT_ROOT, log=print) -> dict[str, Any]:
    if cfg.split not in SPLIT_MANIFESTS:
        raise ValueError(f"split must be one of {sorted(SPLIT_MANIFESTS)}")
    if cfg.figures not in ("all", "failures", "none"):
        raise ValueError("figures must be all | failures | none")
    pred_dir = resolve_path(cfg.pred_dir, project_root)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")
    provenance = resolve_provenance(cfg, pred_dir)
    if not cfg.tag:
        cfg.tag = provenance.get("run_name") or "unnamed_run"
    if any(c in cfg.tag for c in "/\\ "):
        raise ValueError("tag must be a simple name without spaces or slashes")
    enforce_test_protocol(cfg, provenance, project_root)

    manifest = resolve_path(cfg.manifest or SPLIT_MANIFESTS[cfg.split], project_root)
    rows = read_split_manifest(manifest)
    wrong = {r["dataset_split"] for r in rows} - {cfg.split}
    if wrong:
        raise ValueError(f"{manifest} contains other splits {sorted(wrong)}; expected only '{cfg.split}'")
    if cfg.max_cases:
        rows = rows[:cfg.max_cases]

    out_dir = resolve_path(cfg.output_root, project_root) / cfg.tag / cfg.split
    (out_dir / "cases").mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    if cfg.figures != "none":
        (fig_dir / "cases").mkdir(parents=True, exist_ok=True)
        from src.module1_segmentation.evaluation import visualize as V

    missing = [r["scan_id"] for r in rows
               if not (pred_dir / cfg.pred_pattern.format(**r)).is_file()]
    if missing and not cfg.allow_missing:
        raise FileNotFoundError(f"{len(missing)} of {len(rows)} predictions missing in {pred_dir}, e.g. "
                                f"{cfg.pred_pattern.format(**next(r for r in rows if r['scan_id'] == missing[0]))}. "
                                "Run scripts/predict_split.py, or pass --allow-missing for a partial check.")
    train_val = read_reference_dice(cfg.reference_dice_csv)

    records, t_start = [], time.time()
    todo = [r for r in rows if r["scan_id"] not in set(missing)]
    for k, row in enumerate(todo, 1):
        t0 = time.time()
        pred_path = pred_dir / cfg.pred_pattern.format(**row)
        pred_sha = sha256_file(pred_path)
        cache_path = out_dir / "cases" / f"{row['scan_id']}.json"
        rec = _load_cached(cache_path, pred_sha) if cfg.resume else None
        vols = None
        if rec is None:
            vols = load_case(resolve_path(row["mask_path"], project_root), pred_path,
                             label_space=cfg.pred_labels, allow_resample=not cfg.strict_grid)
            result = M.evaluate_volumes(vols.gt, vols.pred, vols.affine, max_shift=cfg.max_shift)
            rec = _case_record(row, result, vols, pred_path, pred_sha, project_root)
            source = "scored"
        else:
            source = "cached"
        case = _restore_nan(rec["case"])
        case["train_val_mean_dice"] = train_val.get(row["scan_id"], float("nan"))
        case["delta_vs_train_val"] = case["mean_dice"] - case["train_val_mean_dice"]

        want_fig = cfg.figures == "all" or (cfg.figures == "failures" and case["failure_category"] != "ok")
        fig_path = fig_dir / "cases" / f"{row['scan_id']}.png"
        if want_fig:
            if not fig_path.is_file() or source == "scored":
                if vols is None:
                    vols = load_case(resolve_path(row["mask_path"], project_root), pred_path,
                                     label_space=cfg.pred_labels, allow_resample=not cfg.strict_grid)
                ct = load_ct_for_display(resolve_path(row["ct_path"], project_root), vols.ref_img)
                V.render_case_figure(ct, vols.gt, vols.pred, vols.affine, fig_path, case)
            case["figure"] = fig_path.relative_to(out_dir).as_posix()
        else:
            case["figure"] = ""
        rec["case"] = case
        _write_json(cache_path, rec)
        records.append(rec)
        log(f"[{k}/{len(todo)}] {row['scan_id']}: {source}, Dice {case['mean_dice']:.3f}, "
            f"spine {case['spine_dice']:.3f}, {case['failure_category']} ({time.time() - t0:.1f}s)")
        del vols

    if not records:
        raise RuntimeError("No case was evaluated")
    cases = [_restore_nan(r["case"]) for r in records]
    levels = [_restore_nan(lv) for r in records for lv in r["levels"]]
    confusion = np.sum([np.asarray(r["confusion"], dtype=np.int64) for r in records], axis=0)
    summary = summarize(cases, levels, confusion)
    assignment = level_assignment(levels)

    _write_csv(out_dir / "per_case.csv", cases, CASE_COLUMNS)
    _write_csv(out_dir / "per_level.csv", levels, LEVEL_COLUMNS)
    _write_csv(out_dir / "summary_per_level.csv", summary["per_level"], list(summary["per_level"][0]))
    failures = sorted([c for c in cases if c["failure_category"] != "ok"],
                      key=lambda c: np.nan_to_num(c["mean_dice"], nan=-1.0))
    _write_csv(out_dir / "failure_cases.csv", failures, CASE_COLUMNS)
    with (out_dir / "confusion_voxels.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference \\ predicted", *M.CLASS_NAMES])
        for name, row in zip(M.CLASS_NAMES, confusion):
            w.writerow([name, *row.tolist()])
    with (out_dir / "level_assignment.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reference \\ assigned", *M.LEVEL_NAMES, "missed"])
        for name, row in zip(M.LEVEL_NAMES, assignment):
            w.writerow([name, *row.tolist()])

    if cfg.figures != "none":
        V.plot_dice_per_level(levels, fig_dir / "dice_per_level.png", cfg.split)
        V.plot_level_assignment(assignment, fig_dir / "level_assignment.png", cfg.split)
        V.plot_spine_vs_level_dice(cases, fig_dir / "spine_vs_level_dice.png", cfg.split)

    meta = {"tag": cfg.tag, "split": cfg.split, "created_utc": _now(), "git_commit": _git_commit(),
            "n_cases": len(cases), "n_missing": len(missing), "missing": missing,
            "n_grid_resampled": sum(int(c["grid_resampled"]) for c in cases),
            "manifest": portable_path(manifest, project_root), "prediction_dir": portable_path(pred_dir, project_root),
            "provenance": provenance, "metrics_version": METRICS_VERSION, "config": asdict(cfg),
            "thresholds": {"missed_coverage": M.MISSED_COVERAGE, "id_distance_mm": M.ID_DISTANCE_MM,
                           "poor_spine_dice": M.POOR_SPINE_DICE, "shift_gain": M.SHIFT_GAIN,
                           "ok_mean_dice": M.OK_MEAN_DICE},
            "seconds": round(time.time() - t_start, 1)}
    _write_json(out_dir / "summary.json", {**meta, **summary})
    write_report(out_dir / "report.md", summary, cases, meta)
    log(f"[done] {len(cases)} cases -> {out_dir}")
    return {**meta, **summary, "output_dir": str(out_dir)}
