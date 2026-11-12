"""Member 5: checkpoint loading and model rebuild."""
from __future__ import annotations

from pathlib import Path
import torch

from src.module1_segmentation.model.unet3d import VertebraUNet

NUM_CLASSES = 9  # 0=bg, 1..8 = T10..L5


def load_checkpoint(checkpoint_path, device):
    """Load best.pt and validate required keys."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)

    required = {"model_state", "model_config", "patch_size_zyx",
                "preprocess_fingerprint", "array_order", "label_map"}
    missing = required - ck.keys()
    if missing:
        raise KeyError(f"Checkpoint missing keys: {missing}")

    if ck["array_order"] != "ZYX":
        raise ValueError(f"Expected ZYX, got {ck['array_order']}")

    return ck


def build_model_from_checkpoint(ck, device):
    """Rebuild VertebraUNet with training architecture and load weights."""
    cfg = ck["model_config"]
    if cfg["in_channels"] != 1 or cfg["out_channels"] != NUM_CLASSES:
        raise ValueError(f"Unexpected channels: {cfg}")

    model = VertebraUNet(channels=tuple(cfg["channels"]))
    model.load_state_dict(ck["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def verify_fingerprint(checkpoint_fingerprint, case_metadata):
    """Ensure the case was preprocessed with the same settings as training."""
    # Member 2 uses "config_fingerprint" in metadata.json
    case_fp = case_metadata.get("config_fingerprint")
    if case_fp is None:
        print("[warn] metadata has no config_fingerprint; skipping compatibility check")
        return
    if case_fp != checkpoint_fingerprint:
        raise ValueError(
            f"Fingerprint mismatch:\n  ckpt={checkpoint_fingerprint}\n  case={case_fp}"
        )