"""Member 2: memory-mapped 3D patch datasets and leak-free DataLoaders.

Input .npy arrays: Z,Y,X. Image tensors: [1,D,H,W], int64 masks: [D,H,W].
Do not fit intensity statistics on val/test: fixed HU window is used instead.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import PROJECT_ROOT, resolve_path


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError("Install PyTorch to create Dataset/DataLoader") from exc
    return torch, DataLoader, Dataset


# Resolve torch base class at import; PyTorch is an explicit Member 2 dependency.
try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        """Allows importing patch utility functions without installing PyTorch."""
        pass


MANIFEST_REQUIRED = {"dataset_split", "patient_id", "scan_id", "image_path", "mask_path", "centers_path", "metadata_path", "config_fingerprint"}


def load_manifest(manifest: str | Path, *, project_root: Path = PROJECT_ROOT) -> list[dict[str, str]]:
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"Processed M2 manifest not found: {path}; run scripts/prepare_data.py first")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not MANIFEST_REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Invalid processed manifest: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No processed cases in {path}")
    keys = [(r["dataset_split"], r["patient_id"], r["scan_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate case in processed manifest: {path}")
    fingerprints = {r["config_fingerprint"] for r in rows}
    if len(fingerprints) != 1:
        raise ValueError("Mixed preprocessing configurations in one manifest")
    for r in rows:
        for col in ("image_path", "mask_path", "centers_path", "metadata_path"):
            if not resolve_path(r[col], project_root).is_file():
                raise FileNotFoundError(f"Missing processed file {col}: {r[col]}")
    return rows


def extract_patch(array_zyx: np.ndarray, start_zyx: tuple[int, int, int], size_zyx: tuple[int, int, int], pad_value: int | float = 0) -> np.ndarray:
    """Extract with explicit out-of-volume padding; start may be negative."""
    if array_zyx.ndim != 3 or len(start_zyx) != 3 or len(size_zyx) != 3 or any(d < 1 for d in size_zyx):
        raise ValueError("Expected a 3D array and three positive patch dimensions")
    source_slices, pads = [], []
    for start, length, dim in zip(start_zyx, size_zyx, array_zyx.shape):
        end = start + length
        source_slices.append(slice(max(0, start), min(dim, end)))
        pads.append((max(0, -start), max(0, end - dim)))
    patch = array_zyx[tuple(source_slices)]
    if any(l or r for l, r in pads):
        patch = np.pad(patch, pads, mode="constant", constant_values=pad_value)
    if patch.shape != tuple(size_zyx):
        raise AssertionError(f"Patch shape mismatch {patch.shape} != {size_zyx}")
    return np.array(patch, copy=True)


def sliding_starts(axis_length: int, patch_size: int, overlap: float = 0.25) -> list[int]:
    """Deterministic start positions including final border; exhaustive coverage."""
    if axis_length < 1 or patch_size < 1 or not (0 <= overlap < 1):
        raise ValueError("Invalid axis length / patch size / overlap")
    if axis_length <= patch_size:
        return [0]
    step = max(1, int(round(patch_size * (1.0 - overlap))))
    stops = list(range(0, axis_length - patch_size + 1, step))
    last = axis_length - patch_size
    if stops[-1] != last:
        stops.append(last)
    return stops


@lru_cache(maxsize=4)
def _mmap_arrays(image_path: str, mask_path: str, center_path: str):
    """Readonly per-worker memmaps: no all-patient RAM loading."""
    return (np.load(image_path, mmap_mode="r", allow_pickle=False),
            np.load(mask_path, mmap_mode="r", allow_pickle=False),
            np.load(center_path, mmap_mode="r", allow_pickle=False))


class VertebraPatchDataset(Dataset):
    """Train: class-balanced random patches; val/test: exhaustive fixed grid.

    Call set_epoch(epoch) each training epoch. DataLoader persistent_workers must
    remain False to propagate the epoch to worker subprocesses.
    """

    def __init__(
        self, manifest: str | Path, *, mode: str,
        patch_size_zyx: tuple[int, int, int] = (96, 96, 96),
        patches_per_case: int = 12,
        foreground_probability: float = 0.75,
        validation_overlap: float = 0.25,
        augment: bool = True,
        seed: int = 42,
        project_root: Path = PROJECT_ROOT,
    ):
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be train, val or test")
        if len(patch_size_zyx) != 3 or any(int(x) != x or x < 1 for x in patch_size_zyx):
            raise ValueError("patch_size_zyx must be three positive integer dimensions")
        if patches_per_case < 1 or not 0 <= foreground_probability <= 1:
            raise ValueError("patches_per_case >= 1 and 0<=foreground_probability<=1 required")
        if not 0 <= validation_overlap < 1:
            raise ValueError("validation_overlap must be in [0,1)")
        if mode != "train" and augment:
            augment = False  # No augmentation for validation/test.
        self.rows = load_manifest(manifest, project_root=project_root)
        self.project_root = project_root
        self.mode = mode
        split_code = {"train": "training", "val": "validation", "test": "test"}[mode]
        if any(row["dataset_split"] != split_code for row in self.rows):
            raise ValueError(f"Manifest contains cases outside {split_code} partition")
        self.patch_size = tuple(map(int, patch_size_zyx))
        self.patches_per_case = patches_per_case
        self.foreground_probability = foreground_probability
        self.augment = augment
        self.seed = int(seed)
        self.epoch = 0
        self.fingerprint = self.rows[0]["config_fingerprint"]
        self._paths = []
        for row in self.rows:
            paths = [str(resolve_path(row[field], project_root)) for field in ("image_path", "mask_path", "centers_path")]
            with resolve_path(row["metadata_path"], project_root).open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if metadata["config_fingerprint"] != self.fingerprint or metadata["dataset_split"] != split_code:
                raise ValueError(f"Metadata mismatch in {row['scan_id']}")
            self._paths.append(paths)
        self._grid: list[tuple[int, tuple[int, int, int]]] = []
        if mode != "train":
            for case_idx, row in enumerate(self.rows):
                with resolve_path(row["metadata_path"], project_root).open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                starts = [sliding_starts(n, p, validation_overlap) for n, p in zip(meta["processed_shape_zyx"], self.patch_size)]
                for z in starts[0]:
                    for y in starts[1]:
                        for x in starts[2]:
                            self._grid.append((case_idx, (z, y, x)))

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_case if self.mode == "train" else len(self._grid)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch, _, _ = _torch()
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.mode == "train":
            case_idx = index % len(self.rows)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        else:
            case_idx, start_zyx = self._grid[index]
            rng = None
        image, mask, centers = _mmap_arrays(*self._paths[case_idx])
        if image.ndim != 3 or mask.shape != image.shape or mask.dtype != np.uint8:
            raise ValueError(f"Corrupt processed image/mask for {self.rows[case_idx]['scan_id']}")
        if self.mode == "train":
            if rng.random() < self.foreground_probability and len(centers) > 0:
                # First select class uniformly (not by voxel population), then location.
                classes = np.unique(centers[:, 0])
                chosen_class = int(rng.choice(classes))
                in_class = np.flatnonzero(centers[:, 0] == chosen_class)
                candidate = centers[int(rng.choice(in_class))]
                center_zyx = tuple(map(int, candidate[1:]))
            else:
                center_zyx = tuple(int(rng.integers(n)) for n in image.shape)
            start_zyx = tuple(int(c - p // 2) for c, p in zip(center_zyx, self.patch_size))
        x = extract_patch(image, start_zyx, self.patch_size, pad_value=0.0)
        y = extract_patch(mask, start_zyx, self.patch_size, pad_value=0)
        if rng is not None and self.augment:
            # Left-right reflection in RAS+ x axis. Identical transform for CT/mask.
            if rng.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            # Mild independent CT contrast and brightness augmentation only.
            scale = float(rng.uniform(0.95, 1.05))
            shift = float(rng.uniform(-0.03, 0.03))
            x = np.clip(x.astype(np.float32) * scale + shift, 0, 1)
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int64)
        row = self.rows[case_idx]
        return {
            "image": torch.from_numpy(x[None]),
            "mask": torch.from_numpy(y),
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
            "start_zyx": torch.tensor(start_zyx, dtype=torch.long),
        }


def assert_patient_disjoint(*datasets: VertebraPatchDataset) -> None:
    owner = {}
    for data in datasets:
        for r in data.rows:
            key = r["patient_id"]
            previous = owner.setdefault(key, data.mode)
            if previous != data.mode:
                raise ValueError(f"Patient leakage: {key} in both {previous} and {data.mode}")
    fingerprints = {d.fingerprint for d in datasets}
    if len(fingerprints) != 1:
        raise ValueError("Train/validation preprocessing fingerprints differ")


def build_dataloaders(
    train_manifest: str | Path, val_manifest: str | Path, *,
    patch_size_zyx: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 2,
    num_workers: int = 0,
    patches_per_case: int = 12,
    seed: int = 42,
    validation_overlap: float = 0.25,
    project_root: Path = PROJECT_ROOT,
):
    """Return train_loader, val_loader, train_dataset, val_dataset.

    Member 4 should call train_dataset.set_epoch(epoch) before each training
    epoch. Keep persistent_workers=False so workers receive the updated epoch.
    """
    torch, DataLoader, _ = _torch()
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Invalid batch_size or num_workers")
    train = VertebraPatchDataset(train_manifest, mode="train", patch_size_zyx=patch_size_zyx,
                                 patches_per_case=patches_per_case, seed=seed, project_root=project_root)
    val = VertebraPatchDataset(val_manifest, mode="val", patch_size_zyx=patch_size_zyx,
                               validation_overlap=validation_overlap, augment=False, project_root=project_root)
    assert_patient_disjoint(train, val)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=torch.cuda.is_available(), generator=generator,
                              persistent_workers=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(), persistent_workers=False)
    return train_loader, val_loader, train, val
