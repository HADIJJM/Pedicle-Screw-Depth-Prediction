"""Smoke-test actual processed M2 cases (requires >=1 train and >=1 val case)."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module1_segmentation.data.dataset import build_dataloaders  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=ROOT / "outputs/m2_preprocessing")
    p.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96), metavar=("D", "H", "W"))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    train, val, train_ds, val_ds = build_dataloaders(
        args.manifest_dir / "train.csv", args.manifest_dir / "val.csv",
        patch_size_zyx=tuple(args.patch_size), batch_size=args.batch_size, num_workers=args.workers,
    )
    print(f"Train cases={len(train_ds.rows)}, patches/epoch={len(train_ds)}")
    print(f"Val cases={len(val_ds.rows)}, exhaustive grid patches={len(val_ds)}")
    print(f"Preprocessing fingerprint={train_ds.fingerprint}")
    for name, loader in (("train", train), ("val", val)):
        batch = next(iter(loader))
        x, y = batch["image"], batch["mask"]
        assert x.ndim == 5 and y.ndim == 4 and x.shape[0] == y.shape[0]
        assert tuple(x.shape[2:]) == tuple(y.shape[1:]) == tuple(args.patch_size)
        assert x.dtype.is_floating_point and y.dtype == __import__("torch").int64
        assert 0 <= float(x.min()) <= float(x.max()) <= 1
        assert 0 <= int(y.min()) <= int(y.max()) <= 8
        print(f"{name} image={tuple(x.shape)} {x.dtype}; mask={tuple(y.shape)} {y.dtype}; "
              f"classes={sorted(set(int(v) for v in y.unique().tolist()))}")
    print("PASS: train/val tensors, labels, split integrity, and shapes")


if __name__ == "__main__":
    main()
