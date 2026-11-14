# Segmentation evaluation: baseline_96 / validation

Generated 2026-11-14T14:06:47Z (git n/a). Research prototype, not for clinical use.

- Cases evaluated: **69** (missing predictions: 0, prediction grid resampled: 0)
- Checkpoint: /content/repo/best.pt (sha256 461a2dcfe136, epoch 54)
- Grid: original CT grid of every case (the validation Dice logged during training used the 2 mm grid).

## Headline

| Metric | Value |
|---|---:|
| Mean Dice over levels (same definition as training) | 0.629 |
| Mean IoU over levels | 0.527 |
| Mean of case Dice | 0.629 |
| Median of case Dice | 0.671 |
| Spine Dice (all target vertebrae as one class) | 0.856 |
| Mean Dice if each case's level names were shifted optimally | 0.662 |
| Identification rate (VerSe style, 20 mm) | 0.821 |

A spine Dice well above the per-level Dice means the bone is found but levels are misnamed.

## Per level

| Level | Cases | Dice mean ± SD | Dice median | IoU mean | Correct | Wrong level | Missed | False pos. | ID rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T10 | 62 | 0.555 ± 0.298 | 0.647 | 0.437 | 47 | 8 | 7 | 7 | 0.823 |
| T11 | 66 | 0.633 ± 0.301 | 0.771 | 0.525 | 52 | 14 | 0 | 3 | 0.803 |
| T12 | 64 | 0.651 ± 0.315 | 0.816 | 0.551 | 51 | 12 | 1 | 5 | 0.906 |
| L1 | 66 | 0.708 ± 0.300 | 0.869 | 0.614 | 57 | 9 | 0 | 3 | 0.848 |
| L2 | 65 | 0.681 ± 0.319 | 0.862 | 0.589 | 54 | 11 | 0 | 4 | 0.800 |
| L3 | 65 | 0.648 ± 0.299 | 0.775 | 0.542 | 54 | 11 | 0 | 4 | 0.815 |
| L4 | 65 | 0.613 ± 0.320 | 0.735 | 0.511 | 51 | 13 | 1 | 4 | 0.754 |
| L5 | 60 | 0.541 ± 0.350 | 0.708 | 0.445 | 43 | 12 | 5 | 9 | 0.817 |

## Failure categories

| Category | Cases |
|---|---:|
| no_prediction | 0 |
| poor_segmentation | 3 |
| level_shift | 5 |
| wrong_level_labels | 27 |
| missed_levels | 3 |
| false_positive_levels | 6 |
| low_overlap | 2 |
| ok | 23 |

## Best uniform level shift per case

Shift +1 = the model named each vertebra one level too cranially (true L3 called L2).

| Shift | Cases |
|---:|---:|
| -1 | 5 |
| +0 | 60 |
| +1 | 4 |

## Ten lowest cases

| Case | Category | Dice | Spine Dice | Best shift → Dice | Missed | Wrong level | Figure |
|---|---|---:|---:|---:|---:|---:|---|
| sub-verse707 | level_shift | 0.006 | 0.819 | +1 → 0.751 | 0 | 6 | [png](figures/cases/sub-verse707.png) |
| sub-verse645 | poor_segmentation | 0.144 | 0.602 | +0 → 0.144 | 0 | 0 | [png](figures/cases/sub-verse645.png) |
| sub-verse601 | level_shift | 0.159 | 0.811 | -1 → 0.405 | 1 | 6 | [png](figures/cases/sub-verse601.png) |
| sub-verse531 | level_shift | 0.210 | 0.849 | -1 → 0.608 | 1 | 5 | [png](figures/cases/sub-verse531.png) |
| sub-verse767 | poor_segmentation | 0.236 | 0.599 | +0 → 0.236 | 0 | 0 | [png](figures/cases/sub-verse767.png) |
| sub-verse757 | poor_segmentation | 0.238 | 0.689 | +0 → 0.238 | 1 | 0 | [png](figures/cases/sub-verse757.png) |
| sub-verse606 | wrong_level_labels | 0.299 | 0.842 | +1 → 0.445 | 0 | 5 | [png](figures/cases/sub-verse606.png) |
| sub-verse615 | level_shift | 0.300 | 0.834 | -1 → 0.518 | 1 | 5 | [png](figures/cases/sub-verse615.png) |
| sub-verse508_dir-ax | level_shift | 0.318 | 0.849 | -1 → 0.517 | 0 | 5 | [png](figures/cases/sub-verse508_dir-ax.png) |
| sub-verse623 | wrong_level_labels | 0.320 | 0.817 | +1 → 0.450 | 1 | 5 | [png](figures/cases/sub-verse623.png) |

## Definitions

- Dice/IoU per level on the reference grid; a level absent in both masks is excluded, a level present in only one mask scores 0.
- Outcome per reference level: *missed* if < 50% of its voxels are covered by any predicted vertebra; otherwise *correct* or *wrong_level* by the label covering most of it.
- Failure category, first rule that matches: no_prediction; poor_segmentation (spine Dice < 0.7); level_shift (a non-zero shift raises mean Dice by ≥ 0.15); wrong_level_labels; missed_levels; false_positive_levels; low_overlap (mean Dice < 0.7); ok.
- Identification: predicted centroid within 20 mm of the reference centroid of the same level, and that centroid is the nearest reference centroid.
