# AI-Assisted Pedicle Screw Depth Prediction

Research prototype for CT-based vertebral segmentation (T10–L5), anatomical analysis, screw-depth estimation, boundary checks, and visualization. **Not for clinical use.**

## Project status

| Module | Scope | Status |
| --- | --- | --- |
| 1 — Data preparation | VerSe2020 inspection, technical validation, official split manifests | Completed |
| 1 — Segmentation model | Preprocessing, 3D segmentation training and evaluation | Not yet implemented |
| 2 | Anatomical and HU analysis | Planned |
| 3 | Screw-depth estimation | Planned |
| 4 | Anatomical constraints and checks | Planned |
| 5 | Visualization | Planned |

The repository may contain placeholder scripts for later stages; placeholders are not working implementations.

## Dataset

Module 1 uses the **VerSe2020** CT scans, vertebra segmentation masks, and centroid JSON files. Data must be obtained separately under the dataset's applicable terms and placed at:

```text
data/raw/VerSe2020/
├── dataset-01training/
├── dataset-02validation/
└── dataset-03test/
```

Each split should retain its original `rawdata/` and `derivatives/` subdirectories. The inspection script discovers `*_ct.nii.gz` files in `rawdata/` and matches vertebra masks and centroid JSON files under `derivatives/`.

**Do not commit raw CT scans, segmentation masks, or other medical-image volumes.**

## Setup

Python 3.10+ is recommended. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The dependencies in `requirements.txt` cover **Module 1 data inspection/validation only**. Future training and visualization components will require additional packages.

## Module 1 — reproducible workflow

Run the following commands from the repository root, in order:

```powershell
# 1. Inventory scans, masks, patient IDs, and vertebral labels
python src/module1_segmentation/data/inspect_dataset.py

# 2. Check technical consistency of CT/mask pairs and metadata
python src/module1_segmentation/data/validate.py

# 3. Preserve the official dataset splits and generate eligible manifests
python src/module1_segmentation/data/split.py

# 4. Verify generated reports and patient separation
python -m unittest discover -s tests/module1 -p "test_*.py" -v
```

`inspect_dataset.py` caches inspected cases; `validate.py` resumes from its saved report and runs pending cases in isolated subprocesses to limit memory usage. These operations can be slow on compressed medical images; **do not force a fresh inspection if complete reports already exist**. `split.py` reads the reports and does not load CT images or randomly reshuffle patients.

### Outputs and handoff

```text
outputs/
├── m1_inspection/
│   ├── dataset_inventory.csv
│   └── inspection_summary.json
├── m1_validation/
│   ├── validation_report.csv
│   └── validation_summary.json
└── m1_splits/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── excluded.csv
    └── split_summary.json
```

The split manifests retain paths relative to the repository root. The receiving team member must download the VerSe2020 dataset separately and reproduce its expected directory structure before accessing images.

### Verified data-preparation results (6 November 2026)

| Split | Original cases | Eligible cases | Excluded cases |
| --- | ---: | ---: | ---: |
| Training | 61 | 50 | 11 |
| Validation | 80 | 69 | 11 |
| Test | 73 | 60 | 13 |
| **Total** | **214** | **179** | **35** |

- Technical validation: **179 passed, 35 warnings, 0 failed**.
- All 35 warnings indicate `no_target_vertebrae` (no labeled vertebra in the T10–L5 target range).
- Patient overlap across official splits: **0**.
- Report consistency tests: **8/8 passed**.
- Eligible means technically validated and containing at least one target vertebra; it does **not** mean clinically validated.

The source VerSe2020 training/validation/test assignments are preserved. The 35 cases with no target vertebrae remain documented in `excluded.csv` and are omitted from eligible manifests.

## Git and data handling

Keep raw images, generated image volumes, checkpoints, and temporary worker files out of Git. For the Module 1 handoff, **commit the small CSV/JSON manifests and reports listed above** after checking they contain no sensitive information or absolute local paths. Ensure `.gitignore` permits these specific reports; the initial repository template ignores all of `outputs/`.

## Limitations

The checks establish file readability, mask-label consistency, geometry/metadata consistency, and split integrity. They do not establish segmentation-model accuracy, optimal screw depth, clinical safety, or regulatory suitability.
