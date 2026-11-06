"""Member 2: geometry-safe VerSe2020 NIfTI preprocessing.

Clinical/research caution: this is a research-only segmentation preprocessing
component. Input CT/mask affines must already agree (as checked by Member 1).
No label-guided crop is performed; the full FOV is retained for inference.

Source NIfTI and all affine matrices are in nibabel (X,Y,Z) voxel order.
Saved .npy arrays and PyTorch patches are in (Z,Y,X) order.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_TO_CLASS: dict[int, int] = {source: source - 16 for source in range(17, 25)}
CLASS_TO_LEVEL: dict[int, str] = {
    0: "background", 1: "T10", 2: "T11", 3: "T12", 4: "L1",
    5: "L2", 6: "L3", 7: "L4", 8: "L5",
}
VERSION = "m2-preprocessing-v1"


@dataclass(frozen=True)
class PreprocessSettings:
    # All values are provisional research defaults; benchmark and agree with M3.
    spacing_xyz_mm: tuple[float, float, float] = (2.0, 2.0, 2.0)
    hu_min: float = -300.0
    hu_max: float = 1500.0
    center_stride: int = 3
    centers_per_class: int = 512
    center_seed: int = 42

    def __post_init__(self) -> None:
        if len(self.spacing_xyz_mm) != 3 or any(
            not np.isfinite(s) or s <= 0 for s in self.spacing_xyz_mm
        ):
            raise ValueError("spacing_xyz_mm must contain three positive finite values")
        if not (np.isfinite(self.hu_min) and np.isfinite(self.hu_max) and self.hu_max > self.hu_min):
            raise ValueError("HU window must have finite hu_min < hu_max")
        if self.center_stride < 1 or self.centers_per_class < 1:
            raise ValueError("center_stride and centers_per_class must be >= 1")

    def fingerprint(self) -> str:
        payload = {"version": VERSION, "settings": asdict(self), "label_map": SOURCE_TO_CLASS}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def resolve_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value).strip())
    if not str(value).strip():
        raise ValueError("Missing image/mask path in Member 1 manifest")
    return (path if path.is_absolute() else project_root / path).resolve()


def portable_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _nibabel():
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to, resample_to_output
    except ImportError as exc:
        raise ImportError("Install requirements.txt (nibabel and scipy) to read NIfTI files") from exc
    return nib, resample_from_to, resample_to_output


def _load_aligned_pair(ct_path: Path, mask_path: Path):
    nib, _, _ = _nibabel()
    ct = nib.load(str(ct_path))
    mask = nib.load(str(mask_path))
    if ct.ndim != 3 or mask.ndim != 3:
        raise ValueError(f"Expected two 3D volumes: CT {ct.shape}, mask {mask.shape}")
    if ct.shape != mask.shape:
        raise ValueError(f"CT/mask shape mismatch: {ct.shape} versus {mask.shape}")
    if not (np.isfinite(ct.affine).all() and np.isfinite(mask.affine).all()):
        raise ValueError("Non-finite CT/mask affine")
    if not np.allclose(ct.affine, mask.affine, rtol=0.0, atol=1e-3):
        raise ValueError("CT/mask affine mismatch; do not align images by array index")
    if abs(float(np.linalg.det(ct.affine[:3, :3]))) < 1e-9:
        raise ValueError("Singular CT affine")
    return ct, mask


def _resample_ct(ct_img: Any, settings: PreprocessSettings):
    nib, _, resample_to_output = _nibabel()
    # Force float32 *before* linear resampling. Many NIfTI CTs are int16;
    # SciPy can otherwise round interpolated intensities back to integer HU.
    # Reading through dataobj also applies the NIfTI slope/intercept (HU).
    ct_values = np.asarray(ct_img.dataobj, dtype=np.float32)
    if not np.isfinite(ct_values).all():
        raise ValueError("Original CT contains NaN or infinite HU values")
    float_image = nib.Nifti1Image(ct_values, ct_img.affine)
    # World-axis-aligned RAS+ grid. A common output shape/affine defines the
    # identical reference grid used for CT and nearest-neighbour mask.
    return resample_to_output(
        float_image, voxel_sizes=settings.spacing_xyz_mm,
        order=1, mode="constant", cval=settings.hu_min,
    )


def _normalize(image_xyz: np.ndarray, settings: PreprocessSettings) -> np.ndarray:
    if not np.isfinite(image_xyz).all():
        raise ValueError("CT contains NaN or infinite values after resampling")
    clipped = np.clip(image_xyz, settings.hu_min, settings.hu_max)
    return ((clipped - settings.hu_min) / (settings.hu_max - settings.hu_min)).astype(np.float16)


def _map_mask(source_xyz: np.ndarray, *, expected_target_count: int | None = None) -> tuple[np.ndarray, list[int]]:
    if not np.isfinite(source_xyz).all():
        raise ValueError("Segmentation mask contains non-finite values")
    if not np.array_equal(source_xyz, np.rint(source_xyz)):
        raise ValueError("Segmentation mask contains non-integer labels")
    unique = {int(x) for x in np.unique(source_xyz)}
    if not unique.issubset(set(range(29))):
        raise ValueError(f"Unexpected VerSe2020 segmentation labels: {sorted(unique - set(range(29)))}")
    present_source = sorted(unique.intersection(SOURCE_TO_CLASS))
    if expected_target_count is not None and len(present_source) != expected_target_count:
        raise ValueError(
            f"Resampling changed target coverage: expected {expected_target_count} "
            f"levels, found {len(present_source)}: {present_source}"
        )
    target = np.zeros(source_xyz.shape, dtype=np.uint8)
    for source, label in SOURCE_TO_CLASS.items():
        target[source_xyz == source] = label
    return target, present_source


def _sample_centers(mask_zyx: np.ndarray, settings: PreprocessSettings, key: str) -> np.ndarray:
    """Store small class-balanced candidate lists; avoid full-volume scans at each training step.

    A regular subgrid is used for memory-bounded discovery. Candidate centers
    are actual voxels of the requested label (no interpolation/blending).
    """
    seed_digest = hashlib.sha256(key.encode("utf-8")).digest()
    key_seed = int.from_bytes(seed_digest[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([settings.center_seed, key_seed]))
    stride = settings.center_stride
    small = mask_zyx[::stride, ::stride, ::stride]
    candidates = []
    for label in range(1, 9):
        matches = np.argwhere(small == label)
        if not len(matches):
            # Sparse labels are not silently discarded: fall back to full grid.
            matches = np.argwhere(mask_zyx == label)
            if len(matches):
                positions = matches
            else:
                continue
        else:
            positions = matches * stride
        selected = rng.choice(len(positions), size=min(len(positions), settings.centers_per_class), replace=False)
        coords = positions[selected]
        candidates.append(np.column_stack((np.full(len(coords), label, np.int32), coords)).astype(np.int32))
    if not candidates:
        raise ValueError("No T10-L5 target voxels survived preprocessing")
    return np.concatenate(candidates, axis=0)


def preprocess_pair(
    ct_path: str | Path,
    mask_path: str | Path,
    settings: PreprocessSettings,
    *,
    expected_target_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return full-field normalized CT and categorical mask, BOTH in Z,Y,X.

    Original anatomical geometry is preserved in metadata for a future inverse
    resampling step. The source CT is never overwritten or rewritten.
    """
    ct_img, mask_img = _load_aligned_pair(Path(ct_path), Path(mask_path))
    nib, resample_from_to, _ = _nibabel()
    ct_rs = _resample_ct(ct_img, settings)
    mask_rs = resample_from_to(
        mask_img, (ct_rs.shape, ct_rs.affine),
        order=0, mode="constant", cval=0.0,
    )
    # For masks the nearest-neighbor operation preserves discrete IDs.
    ct_xyz = np.asarray(ct_rs.dataobj, dtype=np.float32)
    ct_normalized_xyz = _normalize(ct_xyz, settings)
    del ct_xyz
    mask_xyz = np.asarray(mask_rs.dataobj)
    mask_mapped_xyz, source_labels = _map_mask(mask_xyz, expected_target_count=expected_target_count)
    del mask_xyz

    image_zyx = np.ascontiguousarray(ct_normalized_xyz.transpose(2, 1, 0))
    mask_zyx = np.ascontiguousarray(mask_mapped_xyz.transpose(2, 1, 0))
    if image_zyx.shape != mask_zyx.shape:
        raise AssertionError("Internal CT/mask alignment error")
    if not np.array_equal(np.unique(mask_zyx), np.unique(mask_mapped_xyz)):
        raise AssertionError("Unexpected label loss during axis permutation")
    meta = {
        "version": VERSION,
        "settings": asdict(settings),
        "config_fingerprint": settings.fingerprint(),
        "array_order": "ZYX",
        "nifti_order": "XYZ",
        "raw_shape_xyz": list(map(int, ct_img.shape)),
        "raw_affine_xyz": np.asarray(ct_img.affine, dtype=float).tolist(),
        "raw_orientation": list(nib.aff2axcodes(ct_img.affine)),
        "processed_shape_xyz": list(map(int, ct_rs.shape)),
        "processed_shape_zyx": list(map(int, image_zyx.shape)),
        "processed_affine_xyz": np.asarray(ct_rs.affine, dtype=float).tolist(),
        "processed_orientation": list(nib.aff2axcodes(ct_rs.affine)),
        "source_target_labels": source_labels,
        "target_classes_present": [SOURCE_TO_CLASS[i] for i in source_labels],
        "source_to_class": {str(k): v for k, v in SOURCE_TO_CLASS.items()},
        "class_to_level": {str(k): v for k, v in CLASS_TO_LEVEL.items()},
        "normalization": "clip HU then minmax to [0,1]",
        "outside_source_fill": "hu_min CT; 0 mask",
        "ct_interpolation": "linear",
        "mask_interpolation": "nearest",
        "full_field_retained": True,
    }
    return image_zyx, mask_zyx, meta


def preprocess_ct_only(ct_path: str | Path, settings: PreprocessSettings) -> tuple[np.ndarray, dict[str, Any]]:
    """Same deterministic geometry/intensity transform for unlabeled inference CTs.

    Useful to Member 5; does not look at reference labels or bounding boxes.
    """
    nib, _, _ = _nibabel()
    img = nib.load(str(ct_path))
    if img.ndim != 3:
        raise ValueError("Expected 3D CT")
    rs = _resample_ct(img, settings)
    out = np.ascontiguousarray(_normalize(np.asarray(rs.dataobj, dtype=np.float32), settings).transpose(2, 1, 0))
    metadata = {
        "array_order": "ZYX", "raw_shape_xyz": list(img.shape),
        "raw_affine_xyz": np.asarray(img.affine, dtype=float).tolist(),
        "processed_shape_xyz": list(rs.shape),
        "processed_affine_xyz": np.asarray(rs.affine, dtype=float).tolist(),
        "config_fingerprint": settings.fingerprint(),
    }
    return out, metadata


def _clean_component(name: str) -> str:
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {".", ".."}:
        raise ValueError(f"Unsafe or empty patient/scan ID: {name!r}")
    return name


def write_case(
    row: dict[str, str],
    output_root: Path,
    settings: PreprocessSettings,
    *,
    project_root: Path = PROJECT_ROOT,
    overwrite: bool = False,
) -> dict[str, str]:
    """Prepare one case atomically; safe to resume and skip matching settings."""
    split = str(row["dataset_split"]).strip()
    if split not in ("training", "validation", "test"):
        raise ValueError(f"Unknown official split: {split!r}")
    patient = _clean_component(str(row["patient_id"]).strip())
    scan = _clean_component(str(row["scan_id"]).strip())
    key = f"{split}/{patient}/{scan}"
    target_count = int(row["target_count"])
    if target_count < 1 or target_count > 8 or row.get("validation_status") != "passed":
        raise ValueError(f"Case is not Member 1 eligible: {key}")
    source_ct = resolve_path(row["ct_path"], project_root)
    source_mask = resolve_path(row["mask_path"], project_root)
    folder = output_root.resolve() / split / f"{patient}__{scan}"
    meta_path = folder / "metadata.json"
    expected_files = [folder / n for n in ("image.npy", "mask.npy", "centers.npy", "metadata.json")]
    if not overwrite and all(p.is_file() for p in expected_files):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("config_fingerprint") != settings.fingerprint():
            raise ValueError(f"Existing case uses DIFFERENT settings: {key}; use --force to rebuild")
        if meta.get("source_ct") != portable_path(source_ct, project_root) or meta.get("source_mask") != portable_path(source_mask, project_root):
            raise ValueError(f"Source path changed for cached case {key}; use --force to rebuild")
        return _manifest_row(row, folder, meta, project_root)
    if not source_ct.is_file() or not source_mask.is_file():
        raise FileNotFoundError(
            f"Missing original VerSe2020 CT/mask for {key}:\nCT: {source_ct}\nMASK: {source_mask}\n"
            "Member 1 CSV files contain file PATHS, not the actual image volumes."
        )
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{patient}_m2_", dir=folder.parent))
    try:
        image, mask, meta = preprocess_pair(source_ct, source_mask, settings, expected_target_count=target_count)
        centers = _sample_centers(mask, settings, key)
        meta.update(
            patient_id=patient, scan_id=scan, dataset_split=split,
            source_ct=portable_path(source_ct, project_root),
            source_mask=portable_path(source_mask, project_root),
            candidate_centers=int(len(centers)),
        )
        np.save(staging / "image.npy", image, allow_pickle=False)
        np.save(staging / "mask.npy", mask, allow_pickle=False)
        np.save(staging / "centers.npy", centers, allow_pickle=False)
        (staging / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        if folder.exists():
            if not overwrite:
                # Incomplete old cache: replace rather than accidentally treat it as valid.
                shutil.rmtree(folder)
            else:
                shutil.rmtree(folder)
        os.replace(staging, folder)
        return _manifest_row(row, folder, meta, project_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _manifest_row(row: dict[str, str], folder: Path, meta: dict, project_root: Path) -> dict[str, str]:
    return {
        "dataset_split": row["dataset_split"],
        "patient_id": row["patient_id"], "scan_id": row["scan_id"],
        "image_path": portable_path(folder / "image.npy", project_root),
        "mask_path": portable_path(folder / "mask.npy", project_root),
        "centers_path": portable_path(folder / "centers.npy", project_root),
        "metadata_path": portable_path(folder / "metadata.json", project_root),
        "shape_zyx": json.dumps(meta["processed_shape_zyx"]),
        "classes_present": json.dumps(meta["target_classes_present"]),
        "config_fingerprint": meta["config_fingerprint"],
    }
