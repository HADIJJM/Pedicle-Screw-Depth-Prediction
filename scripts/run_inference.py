"""Member 5 CLI: segment T10..L5 using Member 4's best.pt."""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.module1_segmentation.inference.checkpoint import (
    load_checkpoint, build_model_from_checkpoint, verify_fingerprint,
)
from src.module1_segmentation.inference.sliding_window import (
    sliding_window_inference,
)
from src.module1_segmentation.inference.save_output import (
    save_mask_two_spaces, verify_alignment,
)


def resolve_device(name):
    """Pick cuda if available, cpu otherwise; raise if cuda requested but missing."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-npy", required=True, type=Path)
    p.add_argument("--metadata-json", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-original", required=True, type=Path)
    p.add_argument("--output-processed", type=Path, default=None)
    p.add_argument("--reference-ct", type=Path, default=None)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"[info] device: {device}")

    # 1) model first — fail fast if checkpoint is wrong
    t0 = time.time()
    ck = load_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(ck, device)
    print(f"[info] model loaded (epoch={ck['epoch']}, "
          f"val_mean_dice={ck['val_mean_dice']:.4f})")

    # 2) preprocessed case (float16 on disk -> float32 for the model)
    volume_zyx = np.load(args.image_npy).astype(np.float32, copy=False)
    with args.metadata_json.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"[info] volume ZYX = {volume_zyx.shape}")

    # 3) preprocessing compatibility check
    verify_fingerprint(ck["preprocess_fingerprint"], metadata)

    # 4) run the model
    mask_zyx = sliding_window_inference(
        volume_zyx, model, tuple(ck["patch_size_zyx"]),
        device, overlap=args.overlap,
    )
    print(f"[info] inference {time.time() - t0:.1f}s; "
          f"classes = {sorted(np.unique(mask_zyx).tolist())}")

    # 5) save in both spaces
    save_mask_two_spaces(
        mask_zyx, metadata,
        output_original=args.output_original,
        output_processed=args.output_processed,
    )

    # 6) alignment sanity check — auto-resolve raw CT from metadata if not given
    ref_ct = args.reference_ct
    if ref_ct is None and "source_ct" in metadata:
        candidate = ROOT / metadata["source_ct"]
        if candidate.is_file():
            ref_ct = candidate
            print(f"[info] using source_ct from metadata: {ref_ct}")
    if ref_ct is not None:
        verify_alignment(args.output_original, ref_ct)

    print("[done]")


if __name__ == "__main__":
    main()