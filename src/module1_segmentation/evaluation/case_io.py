"""Load reference / predicted masks and put them on ONE spatial grid.

Reference masks are the original VerSe2020 *_seg-vert_msk.nii.gz files listed in
the split manifests (outputs/m1_splits); their labels 17..24 are mapped to classes 1..8
with the preprocessing SOURCE_TO_CLASS so that every non-target label becomes 0, exactly
as during training. Predictions are the inference step's original-space NIfTI masks (0..8).

Comparison happens on the reference grid. If a prediction is on another grid,
it is resampled with nearest-neighbour (never interpolated) and the case is
flagged, because the inference step is supposed to deliver the original grid already.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.module1_segmentation.data.preprocess import SOURCE_TO_CLASS

VERSE_MAX_LABEL = 28
AFFINE_ATOL_MM = 1e-3  # same tolerance preprocessing uses for CT/mask agreement

_VERSE_LUT = np.zeros(256, dtype=np.uint8)
for _src, _cls in SOURCE_TO_CLASS.items():
    _VERSE_LUT[_src] = _cls


def _nib():
    try:
        import nibabel as nib
        import nibabel.processing  # noqa: F401  (registers nib.processing)
    except ImportError as exc:
        raise ImportError("Install requirements.txt (nibabel, scipy) to read NIfTI masks") from exc
    return nib


def _integer_labels(img, path: Path) -> np.ndarray:
    proxy = img.dataobj
    unscaled = (np.issubdtype(img.get_data_dtype(), np.integer) and hasattr(proxy, "get_unscaled")
                and float(getattr(proxy, "slope", 1.0)) == 1.0 and float(getattr(proxy, "inter", 0.0)) == 0.0)
    # Integer masks without scaling are read as stored (no float64 copy of a 100M-voxel volume).
    data = np.asarray(proxy.get_unscaled()) if unscaled else np.asanyarray(proxy)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask: {path} has shape {data.shape}")
    if not np.issubdtype(data.dtype, np.integer):
        if not np.isfinite(data).all():
            raise ValueError(f"Mask contains NaN/inf values: {path}")
        rounded = np.rint(data)
        if not np.array_equal(rounded, data):
            raise ValueError(f"Mask contains non-integer labels: {path}")
        data = rounded
    lo, hi = int(data.min()), int(data.max())
    if lo < 0 or hi > 255:
        raise ValueError(f"Mask labels outside 0..255 in {path}: {lo}..{hi}")
    return data.astype(np.uint8, copy=False)


def map_verse_labels(labels: np.ndarray) -> np.ndarray:
    """VerSe labels (0..28) -> classes 0..8 (17..24 -> 1..8, everything else -> 0)."""
    hi = int(labels.max()) if labels.size else 0
    if hi > VERSE_MAX_LABEL:
        raise ValueError(f"Unexpected VerSe2020 label {hi} (max is {VERSE_MAX_LABEL})")
    return _VERSE_LUT[labels]


def load_reference(path: str | Path):
    """Reference mask -> (classes uint8 XYZ, nibabel image)."""
    nib = _nib()
    path = Path(path)
    img = nib.load(str(path))
    return map_verse_labels(_integer_labels(img, path)), img


def load_prediction(path: str | Path, label_space: str = "model"):
    """Prediction -> (classes uint8 XYZ, nibabel image). label_space: 'model' (0..8) or 'verse'."""
    nib = _nib()
    path = Path(path)
    img = nib.load(str(path))
    labels = _integer_labels(img, path)
    if label_space == "verse":
        return map_verse_labels(labels), img
    if label_space != "model":
        raise ValueError("label_space must be 'model' or 'verse'")
    hi = int(labels.max())
    if hi > 8:
        hint = " It looks like VerSe labels; use --pred-labels verse." if hi <= VERSE_MAX_LABEL else ""
        raise ValueError(f"Prediction labels must be 0..8 (model classes), found max {hi}: {path}.{hint}")
    return labels, img


def grid_difference(ref_img, other_img) -> dict:
    same_shape = tuple(ref_img.shape[:3]) == tuple(other_img.shape[:3])
    diff = float(np.abs(np.asarray(ref_img.affine) - np.asarray(other_img.affine)).max())
    return {"same_shape": same_shape, "affine_max_abs_diff": diff,
            "match": same_shape and diff <= AFFINE_ATOL_MM}


def resample_labels_to(labels: np.ndarray, src_img, ref_img) -> np.ndarray:
    """Nearest-neighbour resampling of a label volume onto the reference grid."""
    nib = _nib()
    src = nib.Nifti1Image(labels, np.asarray(src_img.affine))
    out = nib.processing.resample_from_to(src, (tuple(ref_img.shape[:3]), ref_img.affine),
                                          order=0, mode="constant", cval=0)
    return np.rint(np.asanyarray(out.dataobj)).astype(np.uint8)


@dataclass
class CaseVolumes:
    gt: np.ndarray            # uint8 XYZ, classes 0..8, reference grid
    pred: np.ndarray          # uint8 XYZ, classes 0..8, reference grid
    affine: np.ndarray        # reference voxel -> scanner mm
    zooms: tuple              # reference voxel size (mm)
    grid_resampled: bool
    grid_note: str
    ref_img: object           # nibabel image of the reference mask


def load_case(gt_path, pred_path, *, label_space: str = "model", allow_resample: bool = True) -> CaseVolumes:
    gt, gt_img = load_reference(gt_path)
    pred, pred_img = load_prediction(pred_path, label_space)
    grid = grid_difference(gt_img, pred_img)
    resampled, note = False, "same grid"
    if not grid["match"]:
        note = (f"prediction grid differs (same_shape={grid['same_shape']}, "
                f"max|affine diff|={grid['affine_max_abs_diff']:.4g} mm)")
        if not allow_resample:
            raise ValueError(f"{note}: {pred_path}. Inference output must be on the original CT grid.")
        pred = resample_labels_to(pred, pred_img, gt_img)
        resampled, note = True, note + "; resampled nearest-neighbour to reference grid"
    zooms = tuple(float(z) for z in gt_img.header.get_zooms()[:3])
    return CaseVolumes(gt=gt, pred=pred, affine=np.asarray(gt_img.affine, dtype=np.float64),
                       zooms=zooms, grid_resampled=resampled, grid_note=note, ref_img=gt_img)


def load_ct_for_display(ct_path, ref_img) -> np.ndarray | None:
    """CT in HU (float32, XYZ) when it shares the reference grid, else None."""
    nib = _nib()
    ct = nib.load(str(ct_path))
    if not grid_difference(ref_img, ct)["match"]:
        return None
    return np.asarray(ct.dataobj, dtype=np.float32)
