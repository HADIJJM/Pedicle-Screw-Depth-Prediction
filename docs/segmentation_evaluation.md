# Segmentation evaluation (T10–L5)

This step checks how well the model's vertebra masks match the VerSe2020 reference masks, one vertebral level at a time. It scores each case on that case's original CT grid. It then writes tables, figures and a failure analysis.

Research prototype, **not for clinical use**. No other module, split or setting is changed.

## How it works, in short

1. **Predict:** run the trained model on every case of a split and save one mask per case on the original CT grid.
2. **Match:** pair each prediction with its reference mask through the split manifest.
3. **Score:** compute Dice and IoU per level, plus extra measures that explain the errors. The main question is whether the bone was missed or the vertebra was given the wrong name.
4. **Report:** write CSV tables, a markdown report, summary plots and one overlay figure per case.
5. **Final test:** do all of the above once on the test split, after the model is frozen.

## Files

| Component | Location | What it does |
|---|---|---|
| Metrics | `src/module1_segmentation/evaluation/metrics.py` | Confusion matrix, Dice/IoU per level, spine Dice, level-shift analysis, outcomes, identification, fragments |
| Mask I/O | `src/module1_segmentation/evaluation/case_io.py` | Reads masks, maps VerSe labels 17–24 to classes 1–8, checks that both masks share one grid |
| Runner | `src/module1_segmentation/evaluation/evaluate.py` | Loops over a split, resumes, writes tables and report, locks the test split |
| Figures | `src/module1_segmentation/evaluation/visualize.py` | Per-case overlays on CT and summary plots |
| Synthetic cases | `src/module1_segmentation/evaluation/synthetic.py` | Manufactured NIfTI cases for tests (not medical data) |
| Evaluation CLI | `scripts/evaluate_segmentation.py` | Runs the evaluation |
| Batch inference | `scripts/predict_split.py` | Predicts a whole split with the model loaded once and records the checkpoint hash |
| Colab notebook | `notebooks/evaluation_colab.ipynb` | Runs everything on Google Colab, step by step |
| Tests | `tests/module1/test_evaluation.py` | 17 tests on synthetic data |

## Install

```bash
python -m pip install -r requirements.txt -r requirements-member3.txt -r requirements-evaluation.txt
```

`evaluate_segmentation.py` needs only nibabel, numpy, scipy and matplotlib. `predict_split.py` also needs torch and monai.

## Run (from the repository root)

```bash
# 0. Check the code works (no real data needed)
python -m unittest discover -s tests/module1 -p "test_evaluation.py" -v
python scripts/evaluate_segmentation.py --synthetic

# 1. Predict the validation split
python scripts/predict_split.py --split validation --checkpoint PATH/TO/best.pt --run-name baseline_96 --device cuda
#    -> outputs/predictions/baseline_96/validation/<scan_id>_pred.nii.gz + predictions_manifest.json

# 2. Evaluate the validation split (as often as needed)
python scripts/evaluate_segmentation.py --split validation \
    --pred-dir outputs/predictions/baseline_96/validation \
    --reference-dice outputs/m4_training/m4_runs/baseline_96/case_dice_best.csv
#    -> outputs/evaluation/baseline_96/validation/

# 3. ONCE, after the model is frozen
python scripts/predict_split.py --split test --final-test --checkpoint PATH/TO/best.pt --run-name baseline_96 --device cuda
python scripts/evaluate_segmentation.py --split test --final-test --pred-dir outputs/predictions/baseline_96/test
```

On Colab, `notebooks/evaluation_colab.ipynb` runs the same steps and keeps data and results in Google Drive.

Both scripts are resumable. A prediction is skipped if it already exists for the same checkpoint. An evaluated case is reused while its prediction file is unchanged; this is checked with the file's sha256.

Useful options:

| Option | Effect |
|---|---|
| `--figures failures` | draws case figures only for cases not categorised `ok` (faster) |
| `--pred-labels verse` | for masks that use VerSe labels instead of classes 0–8 |
| `--strict-grid` | fails instead of resampling a prediction that is not on the CT grid |
| `--allow-missing` | allows a partial check; refused on the test split |

## Inputs and matching

* **References:** reference masks and CTs come from the split manifests `outputs/m1_splits/{val,test}.csv`, columns `ct_path` and `mask_path`.
* **Label mapping:** reference labels are mapped with the preprocessing `SOURCE_TO_CLASS` (17..24 → 1..8). Every other label becomes 0 (background), exactly as in training. This includes T13 (28), L6 (25) and the cervical/upper thoracic levels.
* **Predictions:** matched by `scan_id` with the file pattern `{scan_id}_pred.nii.gz`. They must contain classes 0–8.
* **Grid:** scoring happens on the reference grid. If a prediction is off-grid, it is resampled with nearest-neighbour and flagged in `grid_resampled`.

## Outputs (`outputs/evaluation/<run>/<split>/`)

| File | Content |
|---|---|
| `per_case.csv` | one row per case: mean Dice/IoU, spine Dice, best shift, outcome counts, identification rate, category, Dice per level, difference to the training-time validation Dice |
| `per_level.csv` | one row per case and level: status, outcome, Dice, IoU, voxels, volumes (mL), assigned label, centroid distance, components |
| `summary_per_level.csv` | per level: mean, SD, median, quartiles, min/max of Dice and IoU; outcome counts; identification rate |
| `failure_cases.csv` | cases not `ok`, worst first |
| `level_assignment.csv` | cases counted by reference level × the label the prediction mostly gave it |
| `confusion_voxels.csv` | summed 9×9 voxel confusion matrix |
| `summary.json`, `report.md` | headline numbers, per-level table, categories, shifts, ten lowest cases, provenance |
| `figures/` | `dice_per_level.png`, `level_assignment.png`, `spine_vs_level_dice.png`, `cases/<scan_id>.png` |

Git keeps the tables, reports, summary plots and `outputs/evaluation/test_lock.json`. It ignores per-case figures, because they show CT slices, and per-case caches.

## Metric definitions

* **Dice / IoU per level:** a level absent in both masks is excluded from means. A level present in only one mask scores 0. These are the same rules as the validation Dice in `training/train.py`; a test checks that both give identical numbers.
* **Mean Dice over levels:** the mean, over the 8 levels, of each level's mean across cases. This is the same definition as `val_mean_dice` in training. Training computed it on the 2 mm grid, while this evaluation uses the original grid, so small differences are expected.
* **Spine Dice:** all eight target levels merged into one class. A high spine Dice together with a low level Dice means the bone was found but named wrongly.
* **Outcome per reference level:**
  * *missed*: less than 50 % of the level's voxels are covered by any predicted vertebra;
  * otherwise *correct* or *wrong_level*, depending on which label covers most of it;
  * a predicted level that is absent from the reference is a *false_positive*.
* **Best shift:** the single renaming p → p+s (s = −3..3) that maximises a case's mean Dice. A shift of **+1** means every vertebra was named one level too high (towards the head): true L3 called "L2".
* **Identification (VerSe style):** the predicted centroid lies within 20 mm of the reference centroid, and that reference centroid is the nearest one. Centroids come from the masks.
* **Fragments:** the number of 26-connected components for each predicted level.

## Failure categories (first matching rule wins)

| Category | Rule |
|---|---|
| `no_prediction` | no target voxel predicted |
| `poor_segmentation` | spine Dice < 0.70 |
| `level_shift` | a non-zero shift raises mean Dice by ≥ 0.15 |
| `wrong_level_labels` | ≥ 1 reference level mostly given another level's label |
| `missed_levels` | ≥ 1 reference level missed |
| `false_positive_levels` | ≥ 1 level predicted but absent from the reference |
| `low_overlap` | mean Dice < 0.70 |
| `ok` | none of the above |

The thresholds are constants in `metrics.py`, and each run records them in `summary.json`.

A low spine Dice does not always mean poor segmentation. On the validation split, all three `poor_segmentation` cases were upper-spine scans (C7/T1 to T11/T12) with only 2–3 target levels. In those scans the model also predicted lumbar labels.

## Test-split protocol

* `--split test` is refused unless `--final-test` is given.
* The final test needs the checkpoint hash. It comes from `predictions_manifest.json` or from `--checkpoint`.
* The first final test writes `outputs/evaluation/test_lock.json` with that hash.
* Re-running with the same checkpoint is allowed, for example after a plotting fix.
* A different checkpoint is refused. The override `--allow-new-test-checkpoint` is logged in the lock file and must be reported.
* `predict_split.py` also needs `--final-test` for the test split.

## Limitations

* Case figures show one sagittal and one coronal slice: the slices with the most reference vertebra voxels.
* Surface distances (HD95, ASSD) are not computed.
* Dice per level depends on correct naming, so read it together with spine Dice and the shift analysis.
