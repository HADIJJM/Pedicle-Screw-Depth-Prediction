"""Member 4: train the 3D U-Net (T10-L5 vertebra segmentation).

Examples (from the repository root):
  python scripts/train_segmentation.py --synthetic                  # plumbing test, no real data
  python scripts/train_segmentation.py --run-name baseline_96 --device cuda
  python scripts/train_segmentation.py --run-name baseline_96 --resume auto
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.module1_segmentation.data.dataset import _mmap_arrays  # noqa: E402
from src.module1_segmentation.training.train import (  # noqa: E402
    TrainConfig, load_config_file, run_training, with_overrides,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ROOT / "configs/segmentation.yaml")
    p.add_argument("--run-name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--patch-size", type=int, nargs=3, metavar=("D", "H", "W"))
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--patches-per-case", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--val-every", type=int)
    p.add_argument("--device")
    p.add_argument("--output-root")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-train-batches", type=int, help="debug only")
    p.add_argument("--max-val-cases", type=int, help="debug only")
    p.add_argument("--resume", help="'auto' (= <run>/last.pt) or a checkpoint path")
    p.add_argument("--synthetic", action="store_true", help="manufactured arrays; verifies plumbing only")
    a = p.parse_args()

    cfg = load_config_file(a.config) if a.config.is_file() else TrainConfig()
    cfg = with_overrides(
        cfg, run_name=a.run_name, epochs=a.epochs, batch_size=a.batch_size,
        patch_size_zyx=tuple(a.patch_size) if a.patch_size else None, learning_rate=a.learning_rate,
        patches_per_case=a.patches_per_case, num_workers=a.num_workers, val_every=a.val_every,
        device=a.device, output_root=a.output_root, amp=a.amp,
        max_train_batches=a.max_train_batches, max_val_cases=a.max_val_cases)

    if not a.synthetic:
        print(json.dumps(run_training(cfg, resume=a.resume), indent=2, default=str))
        return

    spec = importlib.util.spec_from_file_location("check_member3", ROOT / "scripts/check_member3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as tmp:
        train_csv, val_csv = mod.synthetic_manifests(Path(tmp))
        cfg = with_overrides(
            cfg, train_manifest=str(train_csv), val_manifest=str(val_csv),
            run_name=a.run_name or "synthetic_smoke",
            output_root=a.output_root or str(Path(tmp) / "runs"),
            patch_size_zyx=tuple(a.patch_size) if a.patch_size else (32, 32, 32),
            epochs=a.epochs or 2, val_every=a.val_every or 1,
            patches_per_case=a.patches_per_case or 2, amp=False if a.amp is None else a.amp)
        try:
            result = run_training(cfg, resume=a.resume)
        finally:
            _mmap_arrays.cache_clear()  # release mmap handles (Windows) before temp cleanup
        result["data_kind"] = "synthetic (NOT medical data; proves plumbing only)"
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
