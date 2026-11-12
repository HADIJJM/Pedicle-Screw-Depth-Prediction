"""Member 5: sliding-window inference over a full CT volume."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NUM_CLASSES = 9


class CaseAccumulator:
    """Sums per-patch softmax probabilities into one full-volume prediction."""

    def __init__(self, shape_zyx, num_classes=NUM_CLASSES):
        self.shape = tuple(int(s) for s in shape_zyx)
        self.probs = np.zeros((num_classes, *self.shape), dtype=np.float32)
        self.counts = np.zeros(self.shape, dtype=np.float32)

    def add(self, probs_czyx, start_zyx):
        """Add one patch's probabilities at start_zyx."""
        dst, src = [], []
        for s, n, dim in zip(start_zyx, probs_czyx.shape[1:], self.shape):
            s = int(s)
            lo, hi = max(0, s), min(dim, s + n)
            if hi <= lo:
                return  # patch fully outside volume
            dst.append(slice(lo, hi))
            src.append(slice(lo - s, hi - s))
        self.probs[(slice(None), *dst)] += probs_czyx[(slice(None), *src)]
        self.counts[tuple(dst)] += 1.0

    def prediction(self):
        """Return the uint8 label volume."""
        if (self.counts == 0).any():
            n = int((self.counts == 0).sum())
            raise RuntimeError(f"{n} voxels never covered by a patch")
        return self.probs.argmax(axis=0).astype(np.uint8)


def compute_patch_starts(volume_shape, patch_size, overlap=0.5):
    """List (z,y,x) start coords of all patches covering the volume."""
    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]
    axes = []
    for dim, p, s in zip(volume_shape, patch_size, stride):
        if dim <= p:
            axes.append([0])
        else:
            starts = list(range(0, dim - p + 1, s))
            if starts[-1] + p < dim:
                starts.append(dim - p)  # snap last patch to far edge
            axes.append(starts)
    return [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]


@torch.no_grad()
def sliding_window_inference(volume_zyx, model, patch_size_zyx, device, overlap=0.5):
    """Run model on every patch, return stitched mask with original shape."""
    pz, py, px = patch_size_zyx

    # pad up to patch size if volume is smaller on any axis
    pad = [max(p - d, 0) for p, d in zip(patch_size_zyx, volume_zyx.shape)]
    if any(pad):
        volume_zyx = np.pad(volume_zyx,
                            [(0, pad[0]), (0, pad[1]), (0, pad[2])],
                            mode="constant", constant_values=0.0)

    padded_shape = volume_zyx.shape
    acc = CaseAccumulator(padded_shape)
    coords = compute_patch_starts(padded_shape, patch_size_zyx, overlap)

    for (z, y, x) in coords:
        patch = volume_zyx[z:z+pz, y:y+py, x:x+px]
        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(patch_t)
        probs = F.softmax(logits, dim=1)
        acc.add(probs[0].cpu().numpy(), (z, y, x))

    mask = acc.prediction()

    # strip padding we added
    if any(pad):
        D, H, W = [s - p for s, p in zip(padded_shape, pad)]
        mask = mask[:D, :H, :W]

    return mask