"""Member 4: training loop, validation, checkpoints and experiment records.

Consumes Member 2's DataLoaders and Member 3's VertebraUNet / SegmentationLoss
unchanged. Never reads the test manifest. Research prototype, not for clinical use.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.module1_segmentation.data.dataset import build_dataloaders
from src.module1_segmentation.data.preprocess import PROJECT_ROOT, resolve_path
from src.module1_segmentation.model.losses import SegmentationLoss
from src.module1_segmentation.model.unet3d import LABEL_MAP, VertebraUNet

# ---- case-level validation helpers (patches overlap, so stitch then score per case) ----
NUM_CLASSES = 9  # 0 = background/non-target, 1..8 = T10..L5
LEVEL_NAMES = ("T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5")


def dice_per_class(pred_zyx: np.ndarray, gt_zyx: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Dice for classes 1..num_classes-1. NaN when a class is absent in both masks.

    A class absent in the reference but predicted gives Dice 0 (false positive).
    """
    if pred_zyx.shape != gt_zyx.shape:
        raise ValueError(f"Shape mismatch: pred {pred_zyx.shape} vs gt {gt_zyx.shape}")
    out = np.full(num_classes - 1, np.nan, dtype=np.float64)
    for c in range(1, num_classes):
        p = pred_zyx == c
        g = gt_zyx == c
        ps, gs = int(p.sum()), int(g.sum())
        if ps == 0 and gs == 0:
            continue
        out[c - 1] = 2.0 * int(np.logical_and(p, g).sum()) / (ps + gs)
    return out


class CaseAccumulator:
    """Sums softmax probabilities of overlapping patches for ONE case."""

    def __init__(self, shape_zyx, num_classes: int = NUM_CLASSES):
        self.shape = tuple(int(s) for s in shape_zyx)
        self.probs = np.zeros((num_classes, *self.shape), dtype=np.float32)
        self.counts = np.zeros(self.shape, dtype=np.float32)

    def add(self, probs_czyx: np.ndarray, start_zyx) -> None:
        """Add one patch [C,D,H,W] whose corner is start_zyx (may be negative / overhang)."""
        if probs_czyx.shape[0] != self.probs.shape[0]:
            raise ValueError("Class dimension mismatch")
        dst, src = [], []
        for s, n, dim in zip(start_zyx, probs_czyx.shape[1:], self.shape):
            s = int(s)
            lo, hi = max(0, s), min(dim, s + n)
            if hi <= lo:
                return  # patch lies entirely outside the volume
            dst.append(slice(lo, hi))
            src.append(slice(lo - s, hi - s))
        self.probs[(slice(None), *dst)] += probs_czyx[(slice(None), *src)]
        self.counts[tuple(dst)] += 1.0

    def prediction(self) -> np.ndarray:
        """Argmax label volume uint8 (counts are class-independent, so no division needed)."""
        if (self.counts == 0).any():
            raise ValueError("Some voxels were never covered by a validation patch")
        return self.probs.argmax(axis=0).astype(np.uint8)


def summarize_dice(case_dice: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """(per-class mean over cases where the class exists, mean over classes). NaN-safe."""
    if not case_dice:
        return np.full(NUM_CLASSES - 1, np.nan), float("nan")
    stack = np.vstack(case_dice)
    with np.errstate(all="ignore"):
        valid = ~np.isnan(stack)
        sums = np.where(valid, stack, 0.0).sum(axis=0)
        counts = valid.sum(axis=0)
        per_class = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    finite = per_class[~np.isnan(per_class)]
    return per_class, float(finite.mean()) if finite.size else float("nan")


REGISTRY_COLUMNS = [
    "run_name", "status", "started_utc", "finished_utc", "device", "patch_size_zyx",
    "batch_size", "learning_rate", "epochs_planned", "epochs_done", "best_epoch",
    "best_val_mean_dice", "preprocess_fingerprint", "git_commit",
]


@dataclass(frozen=True)
class TrainConfig:
    run_name: str = "unet3d_baseline"
    train_manifest: str = "outputs/m2_preprocessing/train.csv"
    val_manifest: str = "outputs/m2_preprocessing/val.csv"
    output_root: str = "outputs/m4_training"
    # data (Member 2 interface)
    patch_size_zyx: tuple = (96, 96, 96)
    batch_size: int = 1
    patches_per_case: int = 12
    num_workers: int = 0
    validation_overlap: float = 0.25
    # model (Member 3 interface)
    channels: tuple = (16, 32, 64, 128, 256)
    # optimisation
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    grad_clip_norm: float = 12.0  # 0 disables clipping
    amp: bool = True              # used only on CUDA
    # validation / stopping
    val_every: int = 2
    early_stop_patience: int = 0  # in validation rounds; 0 disables
    # misc
    seed: int = 42
    device: str = "auto"          # auto | cpu | cuda | cuda:N
    log_every: int = 20
    max_train_batches: int = 0    # DEBUG ONLY: 0 = full epoch
    max_val_cases: int = 0        # DEBUG ONLY: 0 = all validation cases

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        data = dict(data)
        for key in ("patch_size_zyx", "channels"):
            if key in data:
                data[key] = tuple(int(v) for v in data[key])
        return cls(**data)

    def validate(self) -> None:
        if len(self.patch_size_zyx) != 3 or any(p < 32 or p % 16 for p in self.patch_size_zyx):
            raise ValueError("patch_size_zyx: three values, each >= 32 and divisible by 16")
        if self.epochs < 1 or self.batch_size < 1 or self.val_every < 1:
            raise ValueError("epochs, batch_size and val_every must be >= 1")
        if self.learning_rate <= 0 or self.min_lr < 0 or self.min_lr > self.learning_rate:
            raise ValueError("Require 0 <= min_lr <= learning_rate and learning_rate > 0")
        if not self.run_name or any(c in self.run_name for c in "/\\ "):
            raise ValueError("run_name must be a simple name without spaces or slashes")
        if "test" in Path(self.train_manifest).name.lower() or "test" in Path(self.val_manifest).name.lower():
            raise ValueError("Test manifests must never be used for training/validation")


def load_config_file(path: str | Path) -> TrainConfig:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install pyyaml: pip install pyyaml") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return TrainConfig.from_dict(data)


# ----------------------------------------------------------------- helpers
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available (no automatic CPU fallback)")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_factor(epoch: int, cfg: TrainConfig) -> float:
    if cfg.warmup_epochs and epoch < cfg.warmup_epochs:
        return (epoch + 1) / cfg.warmup_epochs
    span = max(1, cfg.epochs - cfg.warmup_epochs)
    t = min(1.0, (epoch - cfg.warmup_epochs) / span)
    floor = cfg.min_lr / cfg.learning_rate
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _setup_logger(run_dir: Path, name: str) -> logging.Logger:
    log = logging.getLogger(f"m4.{name}.{id(run_dir)}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def _close_logger(log: logging.Logger) -> None:
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)


def _upsert_registry(root: Path, row: dict[str, Any]) -> None:
    path = root / "experiments.csv"
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_name") != row["run_name"]]
    rows.append({k: row.get(k, "") for k in REGISTRY_COLUMNS})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        w.writeheader()
        w.writerows(rows)


METRIC_COLUMNS = (["epoch", "lr", "train_loss", "val_loss", "val_mean_dice"]
                  + [f"dice_{n}" for n in LEVEL_NAMES] + ["epoch_seconds", "is_best"])


def _truncate_metrics(path: Path, keep_before_epoch: int) -> None:
    """On resume drop rows >= start epoch so a crash never leaves duplicates."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < keep_before_epoch]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    new = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRIC_COLUMNS})


# ------------------------------------------------------------ train / validate
def train_one_epoch(model, criterion, loader, optimizer, scaler, device, cfg, use_amp, epoch, log) -> float:
    model.train()
    total, seen, t0 = 0.0, 0, time.time()
    for i, batch in enumerate(loader):
        if cfg.max_train_batches and i >= cfg.max_train_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss = criterion(logits.float(), masks)  # loss always in float32
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {i}")
        scaler.scale(loss).backward()
        if cfg.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.shape[0]
        seen += images.shape[0]
        if cfg.log_every and (i + 1) % cfg.log_every == 0:
            log.info(f"  epoch {epoch} batch {i + 1}/{len(loader)} loss={total / seen:.4f} "
                     f"({time.time() - t0:.0f}s)")
    if seen == 0:
        raise RuntimeError("Training loader produced no batches")
    return total / seen


@torch.no_grad()
def validate(model, criterion, loader, dataset, device, use_amp, max_cases: int = 0) -> dict[str, Any]:
    """Stitch overlapping validation patches per case and score Dice per level."""
    model.eval()
    rows = {r["scan_id"]: r for r in dataset.rows}
    case_dice: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    state: dict[str, Any] = {"cur": None, "acc": None, "gt": None}
    loss_sum, n = 0.0, 0

    def finish() -> None:
        if state["acc"] is not None:
            case_dice[state["cur"]] = dice_per_class(state["acc"].prediction(), state["gt"])
        state["acc"] = state["gt"] = None

    stop = False
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        loss_sum += criterion(logits, masks).item() * images.shape[0]
        n += images.shape[0]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        starts = batch["start_zyx"].tolist()
        for j, scan in enumerate(batch["scan_id"]):
            if scan != state["cur"]:
                finish()
                if scan in seen:
                    raise RuntimeError(f"Validation patches of {scan} are not contiguous")
                if max_cases and len(case_dice) >= max_cases:
                    stop = True
                    break
                seen.add(scan)
                gt = np.load(resolve_path(rows[scan]["mask_path"], dataset.project_root))
                state.update(cur=scan, gt=gt, acc=CaseAccumulator(gt.shape))
            state["acc"].add(probs[j], starts[j])
        if stop:
            break
    finish()
    per_class, mean_dice = summarize_dice(list(case_dice.values()))
    return {"val_loss": loss_sum / max(n, 1), "mean_dice": mean_dice,
            "per_class": per_class, "case_dice": case_dice}


# ------------------------------------------------------------------ checkpoints
def _model_payload(model, cfg: TrainConfig, epoch: int, result: dict, fingerprint: str, pre_settings) -> dict:
    """Plain tensors/primitives only: loadable with torch.load(weights_only=True) by Member 5."""
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "epoch": epoch,
        "val_mean_dice": float(result["mean_dice"]),
        "val_dice_per_level": {n: (None if np.isnan(v) else float(v))
                               for n, v in zip(LEVEL_NAMES, result["per_class"])},
        "model_config": {"channels": list(cfg.channels), "in_channels": 1, "out_channels": 9},
        "label_map": dict(LABEL_MAP),
        "patch_size_zyx": list(cfg.patch_size_zyx),
        "validation_overlap": cfg.validation_overlap,
        "preprocess_fingerprint": fingerprint,
        "preprocess_settings": pre_settings,
        "array_order": "ZYX",
        "torch_version": str(torch.__version__),
    }


def _rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


# ---------------------------------------------------------------------- main
def run_training(cfg: TrainConfig, resume: str | Path | None = None) -> dict[str, Any]:
    cfg.validate()
    out_root = resolve_path(cfg.output_root, PROJECT_ROOT)
    run_dir = out_root / cfg.run_name
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"

    resume_path: Path | None = None
    if resume is not None:
        resume_path = last_path if str(resume) == "auto" else Path(resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint to resume from not found: {resume_path}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already has files. Use a new --run-name or --resume auto.")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logger(run_dir, cfg.run_name)

    device = _resolve_device(cfg.device)
    use_amp = bool(cfg.amp and device.type == "cuda")
    _set_seed(cfg.seed)
    started = _now()
    status, best_epoch, best_dice, epochs_done = "failed", -1, -1.0, 0
    fingerprint = ""
    try:
        train_loader, val_loader, train_ds, val_ds = build_dataloaders(
            resolve_path(cfg.train_manifest, PROJECT_ROOT), resolve_path(cfg.val_manifest, PROJECT_ROOT),
            patch_size_zyx=cfg.patch_size_zyx, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            patches_per_case=cfg.patches_per_case, seed=cfg.seed, validation_overlap=cfg.validation_overlap)
        fingerprint = train_ds.fingerprint
        meta0 = json.loads(resolve_path(train_ds.rows[0]["metadata_path"], PROJECT_ROOT).read_text(encoding="utf-8"))
        pre_settings = meta0.get("settings")

        model = VertebraUNet(cfg.channels).to(device)
        criterion = SegmentationLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: _lr_factor(e, cfg))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch, no_improve = 0, 0
        if resume_path is not None:
            ck = torch.load(resume_path, map_location="cpu", weights_only=False)  # our own file
            if ck["preprocess_fingerprint"] != fingerprint:
                raise ValueError("Checkpoint was trained with a different preprocessing fingerprint")
            if tuple(ck["patch_size_zyx"]) != tuple(cfg.patch_size_zyx) or list(ck["model_config"]["channels"]) != list(cfg.channels):
                raise ValueError("Checkpoint patch size / model channels differ from the current config")
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            scheduler.load_state_dict(ck["scheduler_state"])
            scaler.load_state_dict(ck["scaler_state"])
            _restore_rng(ck["rng_state"])
            start_epoch = ck["epoch"] + 1
            best_dice, best_epoch, no_improve = ck["best_dice"], ck["best_epoch"], ck["no_improve"]
            _truncate_metrics(metrics_path, start_epoch)
            log.info(f"Resumed from {resume_path} at epoch {start_epoch} (best {best_dice:.4f} @ {best_epoch})")
            _write_json(run_dir / f"config_resume_epoch{start_epoch}.json", asdict(cfg))
        else:
            _write_json(run_dir / "config.json", {
                "config": asdict(cfg), "started_utc": started, "git_commit": _git_commit(),
                "preprocess_fingerprint": fingerprint, "preprocess_settings": pre_settings,
                "train_cases": len(train_ds.rows), "val_cases": len(val_ds.rows),
                "train_patches_per_epoch": len(train_ds), "val_patches": len(val_ds),
                "device": str(device), "amp_active": use_amp,
                "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "command": " ".join(sys.argv)})
        epochs_done = start_epoch
        if start_epoch >= cfg.epochs:
            log.info("Nothing to do: checkpoint already reached the configured number of epochs.")
            status = "completed"
            return {"status": status, "best_epoch": best_epoch, "best_val_mean_dice": best_dice,
                    "epochs_done": epochs_done, "run_dir": str(run_dir)}

        _upsert_registry(out_root, dict(
            run_name=cfg.run_name, status="running", started_utc=started, device=str(device),
            patch_size_zyx="x".join(map(str, cfg.patch_size_zyx)), batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate, epochs_planned=cfg.epochs, epochs_done=epochs_done,
            best_epoch=best_epoch, best_val_mean_dice=best_dice, preprocess_fingerprint=fingerprint,
            git_commit=_git_commit()))
        log.info(f"Run '{cfg.run_name}' on {device} (amp={use_amp}); {len(train_ds.rows)} train / "
                 f"{len(val_ds.rows)} val cases; fingerprint {fingerprint}")

        for epoch in range(start_epoch, cfg.epochs):
            t0 = time.time()
            train_ds.set_epoch(epoch)
            train_loader.generator.manual_seed(cfg.seed + epoch)  # same shuffle after a resume
            train_loss = train_one_epoch(model, criterion, train_loader, optimizer, scaler,
                                         device, cfg, use_amp, epoch, log)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            row: dict[str, Any] = {"epoch": epoch, "lr": f"{lr:.3e}", "train_loss": f"{train_loss:.5f}"}
            is_best, result = False, None
            if (epoch + 1) % cfg.val_every == 0 or epoch + 1 == cfg.epochs:
                result = validate(model, criterion, val_loader, val_ds, device, use_amp, cfg.max_val_cases)
                score = result["mean_dice"]
                score = -1.0 if np.isnan(score) else float(score)
                row.update(val_loss=f"{result['val_loss']:.5f}", val_mean_dice=f"{score:.5f}")
                for name, v in zip(LEVEL_NAMES, result["per_class"]):
                    row[f"dice_{name}"] = "" if np.isnan(v) else f"{v:.5f}"
                if score > best_dice:
                    is_best, best_dice, best_epoch, no_improve = True, score, epoch, 0
                    _atomic_save(_model_payload(model, cfg, epoch, result, fingerprint, pre_settings), best_path)
                    with (run_dir / "case_dice_best.csv").open("w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["scan_id", *[f"dice_{n}" for n in LEVEL_NAMES]])
                        for sid, d in result["case_dice"].items():
                            w.writerow([sid, *["" if np.isnan(x) else f"{x:.5f}" for x in d]])
                else:
                    no_improve += 1
                log.info(f"epoch {epoch}: train={train_loss:.4f} val={result['val_loss']:.4f} "
                         f"dice={score:.4f}{' *best*' if is_best else ''}")
            else:
                log.info(f"epoch {epoch}: train={train_loss:.4f}")

            epochs_done = epoch + 1
            # Checkpoint BEFORE the metrics row so a crash cannot create duplicate rows.
            payload = _model_payload(model, cfg, epoch, result or {"mean_dice": float("nan"), "per_class": [float("nan")] * 8},
                                     fingerprint, pre_settings)
            payload.update(optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                           scaler_state=scaler.state_dict(), rng_state=_rng_state(),
                           best_dice=best_dice, best_epoch=best_epoch, no_improve=no_improve)
            _atomic_save(payload, last_path)
            row.update(epoch_seconds=f"{time.time() - t0:.1f}", is_best=int(is_best))
            _append_metrics(metrics_path, row)

            if cfg.early_stop_patience and no_improve >= cfg.early_stop_patience:
                log.info(f"Early stopping: no improvement for {no_improve} validation rounds")
                break
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        log.info("Interrupted; resume with --resume auto")
        raise
    except Exception as exc:
        log.info(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        summary = {"run_name": cfg.run_name, "status": status, "finished_utc": _now(),
                   "epochs_done": epochs_done, "best_epoch": best_epoch,
                   "best_val_mean_dice": best_dice, "preprocess_fingerprint": fingerprint,
                   "best_checkpoint": str(best_path) if best_path.is_file() else None,
                   "note": "Validation Dice only (val split, stitched per case). Test split untouched."}
        _write_json(run_dir / "summary.json", summary)
        _upsert_registry(out_root, dict(
            run_name=cfg.run_name, status=status, started_utc=started, finished_utc=summary["finished_utc"],
            device=str(device), patch_size_zyx="x".join(map(str, cfg.patch_size_zyx)),
            batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, epochs_planned=cfg.epochs,
            epochs_done=epochs_done, best_epoch=best_epoch, best_val_mean_dice=best_dice,
            preprocess_fingerprint=fingerprint, git_commit=_git_commit()))
        _close_logger(log)
    return {**summary, "run_dir": str(run_dir)}


def with_overrides(cfg: TrainConfig, **overrides: Any) -> TrainConfig:
    return replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
