"""Score predicted vertebra masks per level (T10-L5) and write tables, figures and a report.

Examples (from the repository root):
  python scripts/evaluate_segmentation.py --split validation \
      --pred-dir outputs/predictions/baseline_96/validation \
      --reference-dice outputs/m4_training/m4_runs/baseline_96/case_dice_best.csv
  python scripts/evaluate_segmentation.py --split validation --pred-dir PATH --figures failures
  python scripts/evaluate_segmentation.py --synthetic            # plumbing test, no real data
  # ONCE, after the model is frozen:
  python scripts/evaluate_segmentation.py --split test --final-test \
      --pred-dir outputs/predictions/baseline_96/test
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.module1_segmentation.evaluation.evaluate import (  # noqa: E402
    DEFAULT_PRED_PATTERN, EvalConfig, evaluate_split,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validation", choices=["training", "validation", "test"])
    p.add_argument("--pred-dir", help="folder with predicted masks on the original CT grid")
    p.add_argument("--pred-pattern", default=DEFAULT_PRED_PATTERN,
                   help="file name inside --pred-dir; fields {scan_id} {patient_id} {dataset_split}")
    p.add_argument("--pred-labels", default="model", choices=["model", "verse"],
                   help="model = classes 0..8 (inference default); verse = original labels 17..24")
    p.add_argument("--manifest", help="override the split manifest (default outputs/m1_splits/<split>.csv)")
    p.add_argument("--output-root", default="outputs/evaluation")
    p.add_argument("--tag", help="results folder name (default: run name of the predictions)")
    p.add_argument("--checkpoint", help="best.pt used for the predictions (hashed for provenance)")
    p.add_argument("--figures", default="all", choices=["all", "failures", "none"])
    p.add_argument("--reference-dice", help="training-run case_dice_best.csv to compare per case (2 mm grid)")
    p.add_argument("--allow-missing", action="store_true", help="score only cases with a prediction")
    p.add_argument("--strict-grid", action="store_true", help="fail instead of resampling off-grid predictions")
    p.add_argument("--no-resume", action="store_true", help="re-score cases even if cached")
    p.add_argument("--final-test", action="store_true", help="required for --split test")
    p.add_argument("--allow-new-test-checkpoint", action="store_true",
                   help="override the test lock (breaks the protocol; report it)")
    p.add_argument("--max-cases", type=int, default=0, help="debug only")
    p.add_argument("--synthetic", action="store_true", help="manufactured volumes; verifies plumbing only")
    a = p.parse_args()

    if a.synthetic:
        run_synthetic(a)
        return
    if not a.pred_dir:
        p.error("--pred-dir is required (or use --synthetic)")
    cfg = EvalConfig(pred_dir=a.pred_dir, split=a.split, manifest=a.manifest, pred_pattern=a.pred_pattern,
                     pred_labels=a.pred_labels, output_root=a.output_root, tag=a.tag, checkpoint=a.checkpoint,
                     figures=a.figures, reference_dice_csv=a.reference_dice, allow_missing=a.allow_missing,
                     strict_grid=a.strict_grid, resume=not a.no_resume, final_test=a.final_test,
                     allow_new_test_checkpoint=a.allow_new_test_checkpoint, max_cases=a.max_cases)
    result = evaluate_split(cfg)
    print(json.dumps({"output_dir": result["output_dir"], "n_cases": result["n_cases"],
                      **result["overall"]}, indent=2, default=str))


def run_synthetic(a) -> None:
    """Build fake VerSe-like NIfTI cases (incl. a level shift and an LPS scan) and evaluate them."""
    from src.module1_segmentation.evaluation.synthetic import make_synthetic_split
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest, pred_dir = make_synthetic_split(tmp)
        out_root = Path(a.output_root) if a.output_root != "outputs/evaluation" else tmp / "evaluation"
        cfg = EvalConfig(pred_dir=str(pred_dir), split="validation", manifest=str(manifest),
                         output_root=str(out_root), tag="synthetic_smoke", figures=a.figures)
        result = evaluate_split(cfg, project_root=tmp)
        print(json.dumps({"data_kind": "synthetic (NOT medical data; proves plumbing only)",
                          "output_dir": result["output_dir"], **result["overall"],
                          "failure_categories": result["failure_categories"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
