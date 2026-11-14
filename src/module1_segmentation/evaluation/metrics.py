"""Per-level segmentation metrics for T10-L5 (classes 1..8).

Every overlap metric is derived from ONE 9x9 voxel confusion matrix per case
(rows = reference class, columns = predicted class), so Dice, IoU, the
spine-as-one-class scores, the per-level outcome and the level-shift analysis
all come from a single pass over the volume.

Conventions (identical to the validation Dice in training/train.py, so numbers are comparable):
  * level absent in BOTH reference and prediction -> NaN (excluded from means);
  * level absent in the reference but predicted   -> Dice 0 (false positive);
  * level present in the reference but not predicted -> Dice 0 (missed).

Arrays are label volumes in the same voxel order as the NIfTI they came from
(nibabel X,Y,Z on the original CT grid). Research prototype, not for clinical use.
"""
from __future__ import annotations

from typing import Any

import numpy as np

NUM_CLASSES = 9  # 0 = background / non-target, 1..8 = T10..L5
LEVEL_NAMES = ("T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5")
CLASS_NAMES = ("background",) + LEVEL_NAMES

# A reference level counts as "missed" when less than this fraction of its
# voxels is covered by ANY predicted vertebra label.
MISSED_COVERAGE = 0.5
# VerSe benchmark identification criterion (Sekuboyina et al., MedIA 2021):
# predicted centroid within 20 mm of the reference centroid AND that reference
# centroid is the closest one.
ID_DISTANCE_MM = 20.0
# Failure-category thresholds (documented in docs/segmentation_evaluation.md).
POOR_SPINE_DICE = 0.70
SHIFT_GAIN = 0.15
OK_MEAN_DICE = 0.70
_CHUNK_VOXELS = 8_000_000


# --------------------------------------------------------------------- basics
def _check_labels(name: str, arr: np.ndarray, num_classes: int) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D label volume, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be an integer label volume, got {arr.dtype}")
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi >= num_classes:
        raise ValueError(f"{name} labels must be 0..{num_classes - 1}, found range {lo}..{hi}")


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Voxel counts int64[num_classes, num_classes]; rows = reference, cols = prediction.

    Processed in chunks so a 512x512x400 volume never needs an int64 copy.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: reference {gt.shape} vs prediction {pred.shape}")
    _check_labels("reference", gt, num_classes)
    _check_labels("prediction", pred, num_classes)
    g = gt.reshape(-1)
    p = pred.reshape(-1)
    cm = np.zeros(num_classes * num_classes, dtype=np.int64)
    for start in range(0, g.size, _CHUNK_VOXELS):
        gs = g[start:start + _CHUNK_VOXELS].astype(np.int32)
        ps = p[start:start + _CHUNK_VOXELS].astype(np.int32)
        cm += np.bincount(gs * num_classes + ps, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def dice_iou_from_confusion(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dice and IoU for classes 1..C-1 (NaN when a class is absent in both)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)[1:]
    gt_n = cm.sum(axis=1)[1:]
    pred_n = cm.sum(axis=0)[1:]
    denom = gt_n + pred_n
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(denom > 0, 2.0 * tp / denom, np.nan)
        union = denom - tp
        iou = np.where(union > 0, tp / union, np.nan)
    return dice, iou


def spine_scores(cm: np.ndarray) -> tuple[float, float]:
    """Dice/IoU of 'any target vertebra' vs background: ignores WHICH level was named."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = cm[1:, 1:].sum()
    gt_n = cm[1:, :].sum()
    pred_n = cm[:, 1:].sum()
    if gt_n + pred_n == 0:
        return float("nan"), float("nan")
    return float(2.0 * tp / (gt_n + pred_n)), float(tp / (gt_n + pred_n - tp))


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


# --------------------------------------------------------------- level shifts
def shift_confusion(cm: np.ndarray, shift: int) -> np.ndarray:
    """Confusion matrix after renaming every predicted level p -> p + shift.

    Labels pushed outside 1..8 become background. shift=+1 means the model named
    each vertebra one level too cranially (e.g. true L3 predicted as "L2").
    """
    n = cm.shape[0]
    out = np.zeros_like(cm)
    out[:, 0] += cm[:, 0]
    for p in range(1, n):
        q = p + shift
        out[:, q if 1 <= q < n else 0] += cm[:, p]
    return out


def shift_analysis(cm: np.ndarray, max_shift: int = 3) -> dict[str, Any]:
    """Mean Dice for each uniform renaming shift; best = highest (ties -> smallest |shift|)."""
    by_shift: dict[int, float] = {}
    for s in range(-max_shift, max_shift + 1):
        dice, _ = dice_iou_from_confusion(shift_confusion(cm, s))
        by_shift[s] = nanmean(dice)
    valid = {s: v for s, v in by_shift.items() if not np.isnan(v)}
    if not valid:
        return {"best_shift": 0, "mean_dice_at_best_shift": float("nan"), "mean_dice_by_shift": by_shift}
    best = max(valid, key=lambda s: (round(valid[s], 9), -abs(s)))
    return {"best_shift": int(best), "mean_dice_at_best_shift": float(valid[best]),
            "mean_dice_by_shift": by_shift}


# ------------------------------------------------------- geometry per level
def label_bboxes(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> dict[int, tuple[slice, ...]]:
    """Bounding boxes of classes 1..C-1 in one pass (scipy.ndimage.find_objects)."""
    from scipy import ndimage
    objs = ndimage.find_objects(labels, max_label=num_classes - 1)
    return {c: objs[c - 1] for c in range(1, num_classes) if objs[c - 1] is not None}


def centroids_mm(labels: np.ndarray, affine: np.ndarray, bboxes: dict[int, tuple[slice, ...]]) -> dict[int, np.ndarray]:
    """World-space (scanner mm) centroid of every present class."""
    out = {}
    for c, bb in bboxes.items():
        idx = np.argwhere(labels[bb] == c).mean(axis=0) + np.array([s.start for s in bb], dtype=np.float64)
        out[c] = (affine @ np.append(idx, 1.0))[:3]
    return out


def component_stats(labels: np.ndarray, bboxes: dict[int, tuple[slice, ...]]) -> dict[int, tuple[int, float]]:
    """(number of 26-connected components, fraction of voxels in the largest) per class."""
    from scipy import ndimage
    structure = np.ones((3, 3, 3), dtype=bool)
    out = {}
    for c, bb in bboxes.items():
        comp, n = ndimage.label(labels[bb] == c, structure=structure)
        sizes = np.bincount(comp.reshape(-1))[1:]
        out[c] = (int(n), float(sizes.max() / sizes.sum()) if n else float("nan"))
    return out


# ------------------------------------------------------------------ one case
def evaluate_volumes(
    gt: np.ndarray,
    pred: np.ndarray,
    affine: np.ndarray,
    *,
    max_shift: int = 3,
    id_distance_mm: float = ID_DISTANCE_MM,
) -> dict[str, Any]:
    """All evaluation metrics for one case on ONE grid.

    Returns {"confusion": int64[9,9], "levels": [dict per class 1..8], "case": dict}.
    """
    affine = np.asarray(affine, dtype=np.float64)
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    cm = confusion_matrix(gt, pred)
    dice, iou = dice_iou_from_confusion(cm)
    spine_dice, spine_iou = spine_scores(cm)
    shifts = shift_analysis(cm, max_shift=max_shift)

    gt_boxes, pred_boxes = label_bboxes(gt), label_bboxes(pred)
    gt_cent = centroids_mm(gt, affine, gt_boxes)
    pred_cent = centroids_mm(pred, affine, pred_boxes)
    pred_comp = component_stats(pred, pred_boxes)

    levels = []
    for c in range(1, NUM_CLASSES):
        gt_n, pred_n = int(cm[c, :].sum()), int(cm[:, c].sum())
        status = ("both" if gt_n and pred_n else "gt_only" if gt_n
                  else "pred_only" if pred_n else "absent")
        row: dict[str, Any] = {
            "class": c, "level": LEVEL_NAMES[c - 1], "status": status,
            "gt_voxels": gt_n, "pred_voxels": pred_n, "tp_voxels": int(cm[c, c]),
            "gt_volume_ml": gt_n * voxel_ml, "pred_volume_ml": pred_n * voxel_ml,
            "dice": float(dice[c - 1]), "iou": float(iou[c - 1]),
            "outcome": "absent", "assigned_label": "", "assigned_fraction": float("nan"),
            "pred_sits_on": "", "centroid_distance_mm": float("nan"), "identified": "",
            "pred_components": pred_comp.get(c, (0, float("nan")))[0],
            "largest_component_fraction": pred_comp.get(c, (0, float("nan")))[1],
        }
        if gt_n:
            covered = int(cm[c, 1:].sum())
            if covered < MISSED_COVERAGE * gt_n:
                row.update(outcome="missed", assigned_label="background",
                           assigned_fraction=cm[c, 0] / gt_n)
            else:
                dom = int(np.argmax(cm[c, 1:])) + 1
                row.update(outcome="correct" if dom == c else "wrong_level",
                           assigned_label=CLASS_NAMES[dom], assigned_fraction=cm[c, dom] / gt_n)
        elif pred_n:
            row["outcome"] = "false_positive"
        if pred_n:
            # Which reference structure do the voxels predicted as this level lie on?
            row["pred_sits_on"] = CLASS_NAMES[int(np.argmax(cm[:, c]))]
        if gt_n:
            ok = False
            if c in pred_cent:
                d = float(np.linalg.norm(pred_cent[c] - gt_cent[c]))
                row["centroid_distance_mm"] = d
                nearest = min(gt_cent, key=lambda k: np.linalg.norm(pred_cent[c] - gt_cent[k]))
                ok = d <= id_distance_mm and nearest == c
            row["identified"] = int(ok)
        levels.append(row)

    outcomes = [r["outcome"] for r in levels]
    n_gt = sum(r["gt_voxels"] > 0 for r in levels)
    mean_dice, mean_iou = nanmean(dice), nanmean(iou)
    case = {
        "n_levels_gt": n_gt,
        "n_levels_pred": sum(r["pred_voxels"] > 0 for r in levels),
        "levels_gt": " ".join(r["level"] for r in levels if r["gt_voxels"]),
        "mean_dice": mean_dice, "mean_iou": mean_iou,
        "spine_dice": spine_dice, "spine_iou": spine_iou,
        "best_shift": shifts["best_shift"],
        "mean_dice_at_best_shift": shifts["mean_dice_at_best_shift"],
        "shift_gain": shifts["mean_dice_at_best_shift"] - mean_dice,
        "n_correct": outcomes.count("correct"), "n_wrong_level": outcomes.count("wrong_level"),
        "n_missed": outcomes.count("missed"), "n_false_positive": outcomes.count("false_positive"),
        "n_identified": sum(r["identified"] == 1 for r in levels),
        "id_rate": (sum(r["identified"] == 1 for r in levels) / n_gt) if n_gt else float("nan"),
        "pred_foreground_voxels": int(cm[:, 1:].sum()),
        "fragmented_levels": " ".join(r["level"] for r in levels if r["pred_components"] > 1),
    }
    case["failure_category"] = failure_category(case)
    return {"confusion": cm, "levels": levels, "case": case,
            "mean_dice_by_shift": shifts["mean_dice_by_shift"]}


def failure_category(case: dict[str, Any]) -> str:
    """One primary label per case, first matching rule wins (see docs/segmentation_evaluation.md)."""
    if case["pred_foreground_voxels"] == 0:
        return "no_prediction"
    if not np.isnan(case["spine_dice"]) and case["spine_dice"] < POOR_SPINE_DICE:
        return "poor_segmentation"
    if case["best_shift"] != 0 and case["shift_gain"] >= SHIFT_GAIN:
        return "level_shift"
    if case["n_wrong_level"]:
        return "wrong_level_labels"
    if case["n_missed"]:
        return "missed_levels"
    if case["n_false_positive"]:
        return "false_positive_levels"
    if np.isnan(case["mean_dice"]) or case["mean_dice"] < OK_MEAN_DICE:
        return "low_overlap"
    return "ok"


FAILURE_CATEGORIES = ("no_prediction", "poor_segmentation", "level_shift", "wrong_level_labels",
                      "missed_levels", "false_positive_levels", "low_overlap", "ok")
