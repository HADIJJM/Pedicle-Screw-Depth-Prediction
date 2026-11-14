# Example comparison figures (validation)

Six case figures chosen to illustrate the failure analysis. They are unchanged copies of `figures/cases/<scan_id>.png`, as written by `scripts/evaluate_segmentation.py`. The full set of 69 case figures stays out of Git because the figures show CT slices.

Each figure shows sagittal (top) and coronal (bottom) slices with three panels: reference mask, prediction, and error map. In the error map: green = correct, yellow = missed, orange = wrong level, red = false positive.

| File | Case | Mean Dice | Spine Dice | What it shows |
|---|---|---:|---:|---|
| `01_correct_sub-verse761.png` | sub-verse761 | 0.905 | 0.921 | Best case: all eight levels found and named correctly. |
| `02_six_lumbar_vertebrae_sub-verse707.png` | sub-verse707 | 0.006 | 0.819 | The patient has six lumbar vertebrae; the reference mask contains label 25 (L6). Training maps L6 to background, so the model names the lowest lumbar vertebra "L5" and every level above it is shifted by one. Renaming the levels by one gives Dice 0.751. |
| `03_mixed_level_names_sub-verse623.png` | sub-verse623 | 0.320 | 0.817 | Bone is found, but names are mixed inside one scan: upper levels are shifted by one, L4–L5 are correct, and some vertebrae carry two labels. |
| `04_partly_visible_vertebra_sub-verse640.png` | sub-verse640 | 0.437 | 0.819 | T10–L1 are all correct. The model also labels the partly visible vertebra at the bottom edge, which has no reference label, and a small region in the upper spine. Each extra level scores Dice 0, so the case mean drops. |
| `05_upper_spine_scan_sub-verse645.png` | sub-verse645 | 0.144 | 0.602 | The scan covers C7–T11, so only T10–T11 are in the target range. The model also predicts lower labels in this scan. A low spine Dice here does not mean the bone is segmented poorly. |
| `06_missed_lower_lumbar_sub-verse705.png` | sub-verse705 | 0.641 | 0.750 | Upper levels are correct. L4 and L5 are mostly missed, next to very bright material in the image (likely metal). |
