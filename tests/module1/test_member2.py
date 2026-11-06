"""Member 2 tests with small synthetic volumes; no original patient data needed.

NIfTI integration checks are automatically enabled when nibabel is installed.
"""
from __future__ import annotations

import csv
import gc
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.module1_segmentation.data.preprocess import (
    CLASS_TO_LEVEL, SOURCE_TO_CLASS, PreprocessSettings, _map_mask,
    _sample_centers, preprocess_ct_only, preprocess_pair, resolve_path, write_case,
)
from src.module1_segmentation.data.dataset import (
    VertebraPatchDataset, assert_patient_disjoint, build_dataloaders,
    extract_patch, sliding_starts, _mmap_arrays,
)

NIBABEL_AVAILABLE = importlib.util.find_spec("nibabel") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestCorePreprocessing(unittest.TestCase):
    def test_label_map_matches_verse_convention(self):
        self.assertEqual(SOURCE_TO_CLASS, {i: i - 16 for i in range(17, 25)})
        self.assertEqual(CLASS_TO_LEVEL[1], "T10")
        self.assertEqual(CLASS_TO_LEVEL[8], "L5")
        src = np.array([[[0, 1, 16, 17, 18, 24, 25, 28]]], dtype=np.float32)
        mapped, present = _map_mask(src, expected_target_count=3)
        self.assertEqual(mapped.tolist(), [[[0, 0, 0, 1, 2, 8, 0, 0]]])
        self.assertEqual(present, [17, 18, 24])

    def test_invalid_mask_and_coverage_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "non-integer"):
            _map_mask(np.array([[[17.2]]]))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            _map_mask(np.array([[[36.0]]]))
        with self.assertRaisesRegex(ValueError, "coverage"):
            _map_mask(np.array([[[17.0]]]), expected_target_count=2)
        with self.assertRaises(ValueError):
            PreprocessSettings(spacing_xyz_mm=(0.0, 2.0, 2.0))
        with self.assertRaises(ValueError):
            PreprocessSettings(hu_min=10, hu_max=10)

    def test_class_balanced_centers_accurate(self):
        mask = np.zeros((35, 22, 18), dtype=np.uint8)
        mask[4:18, 4:16, 4:13] = 1
        mask[20:32, 4:16, 4:13] = 8
        settings = PreprocessSettings(center_stride=3, centers_per_class=20)
        c1 = _sample_centers(mask, settings, "caseA")
        c2 = _sample_centers(mask, settings, "caseA")
        np.testing.assert_array_equal(c1, c2)
        self.assertEqual(set(c1[:, 0]), {1, 8})
        self.assertTrue(np.all(mask[c1[:, 1], c1[:, 2], c1[:, 3]] == c1[:, 0]))
        self.assertEqual(len(c1), 40)

    def test_patch_padding_and_border_coverage(self):
        a = np.arange(5 * 6 * 7).reshape(5, 6, 7)
        patch = extract_patch(a, (-2, 4, 5), (4, 4, 4), pad_value=-10)
        self.assertEqual(patch.shape, (4, 4, 4))
        np.testing.assert_array_equal(patch[2:, :2, :2], a[:2, 4:6, 5:7])
        self.assertEqual(patch[0, 0, 0], -10)
        covered = np.zeros((23, 24, 25), dtype=bool)
        starts = [sliding_starts(v, 10, 0.25) for v in covered.shape]
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    covered[z:z + 10, y:y + 10, x:x + 10] = True
        self.assertTrue(covered.all())
        self.assertEqual(sliding_starts(3, 10), [0])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestDatasetWithSyntheticArrays(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Windows does not allow deleting open memory-mapped .npy files.
        # Register this cleanup AFTER temp cleanup, since unittest calls
        # cleanups in LIFO order (cache release occurs before folder removal).
        def release_memmaps():
            _mmap_arrays.cache_clear()
            gc.collect()
        self.addCleanup(release_memmaps)
        self.root = Path(self.tmp.name)
        self.manifests = {}
        for split, filename, patients in (
            ("training", "train.csv", ("patientA", "patientB")),
            ("validation", "val.csv", ("patientC",)),
            ("test", "test.csv", ("patientD",)),
        ):
            rows = []
            for patient in patients:
                case_dir = self.root / split / patient
                case_dir.mkdir(parents=True)
                mask = np.zeros((19, 23, 25), dtype=np.uint8)
                mask[3:11, 3:10, 2:10] = 1
                mask[12:17, 12:20, 13:23] = 8
                image = mask.astype(np.float16) / 8.0
                centers = np.array([[1, 6, 6, 6], [8, 14, 15, 18]], dtype=np.int32)
                for name, arr in (("image.npy", image), ("mask.npy", mask), ("centers.npy", centers)):
                    np.save(case_dir / name, arr)
                metadata = {
                    "processed_shape_zyx": list(mask.shape),
                    "config_fingerprint": "testconfig", "dataset_split": split,
                    "scan_id": patient + "_scan", "patient_id": patient,
                    "array_order": "ZYX",
                    "target_classes_present": [1, 8],
                    "processed_affine_xyz": np.eye(4).tolist(),
                    "raw_affine_xyz": np.eye(4).tolist(),
                }
                (case_dir / "metadata.json").write_text(json.dumps(metadata))
                rows.append({"dataset_split": split, "patient_id": patient, "scan_id": patient + "_scan",
                             "image_path": str(case_dir / "image.npy"), "mask_path": str(case_dir / "mask.npy"),
                             "centers_path": str(case_dir / "centers.npy"),
                             "metadata_path": str(case_dir / "metadata.json"), "config_fingerprint": "testconfig"})
            csv_path = self.root / filename
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            self.manifests[split] = csv_path

    def test_torch_shapes_labels_and_batch(self):
        train, val, train_ds, val_ds = build_dataloaders(
            self.manifests["training"], self.manifests["validation"],
            patch_size_zyx=(12, 12, 12), batch_size=2, patches_per_case=3,
            project_root=self.root,
        )
        self.assertEqual(len(train_ds), 6)
        self.assertGreater(len(val_ds), 1)
        train_batch, val_batch = next(iter(train)), next(iter(val))
        self.assertEqual(tuple(train_batch["image"].shape), (2, 1, 12, 12, 12))
        self.assertEqual(tuple(train_batch["mask"].shape), (2, 12, 12, 12))
        self.assertEqual(tuple(val_batch["image"].shape), (2, 1, 12, 12, 12))
        self.assertTrue(train_batch["mask"].dtype == __import__("torch").int64)
        self.assertLessEqual(int(train_batch["mask"].max()), 8)
        self.assertGreaterEqual(float(train_batch["image"].min()), 0.0)
        self.assertLessEqual(float(train_batch["image"].max()), 1.0)
        assert_patient_disjoint(train_ds, val_ds)

    def test_deterministic_sampling_and_augmentation_geometry(self):
        ds = VertebraPatchDataset(self.manifests["training"], mode="train",
                                  patch_size_zyx=(13, 11, 11), patches_per_case=5,
                                  foreground_probability=1.0, seed=123,
                                  project_root=self.root)
        item1 = ds[0]
        item2 = ds[0]
        self.assertTrue(__import__("torch").equal(item1["image"], item2["image"]))
        self.assertTrue(__import__("torch").equal(item1["mask"], item2["mask"]))
        ds.set_epoch(1)
        different_epoch = ds[0]
        self.assertFalse(__import__("torch").equal(item1["start_zyx"], different_epoch["start_zyx"]) and
                         __import__("torch").equal(item1["image"], different_epoch["image"]))
        # Synthetic image = mapped label/8. Class-specific positives remain spatially aligned
        # even after an optional LR flip and mild image intensity jitter.
        for idx in range(len(ds)):
            item = ds[idx]
            img = item["image"][0].numpy()
            label = item["mask"].numpy()
            self.assertTrue(np.all(img[label > 0] > 0.08))
            self.assertTrue(np.all(img[label == 0] < 0.1))

    def test_val_grid_exactly_covers_full_volume(self):
        ds = VertebraPatchDataset(self.manifests["validation"], mode="val",
                                  patch_size_zyx=(12, 12, 12), augment=True,
                                  project_root=self.root)
        covered = np.zeros((19, 23, 25), dtype=bool)
        for item in ds:
            z, y, x = map(int, item["start_zyx"].tolist())
            covered[z:z + 12, y:y + 12, x:x + 12] = True
        self.assertTrue(covered.all())
        self.assertFalse(ds.augment)
        self.assertEqual(tuple(ds[0]["image"].shape), (1, 12, 12, 12))

    def test_output_cache_validation(self):
        from scripts.validate_prepared_data import validate_case
        for partition in self.manifests:
            from src.module1_segmentation.data.dataset import load_manifest
            for row in load_manifest(self.manifests[partition], project_root=self.root):
                validate_case(row, full=True)

    def test_leakage_fails(self):
        ds1 = VertebraPatchDataset(self.manifests["training"], mode="train", project_root=self.root)
        ds2 = VertebraPatchDataset(self.manifests["validation"], mode="val", project_root=self.root)
        ds2.rows[0]["patient_id"] = ds1.rows[0]["patient_id"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_patient_disjoint(ds1, ds2)


@unittest.skipUnless(NIBABEL_AVAILABLE, "Install nibabel to run real NIfTI affine/resampling integration tests")
class TestRealNiftiIntegration(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        ct = np.full((29, 25, 23), -300, dtype=np.float32)
        mask = np.zeros(ct.shape, dtype=np.int16)
        mask[3:12, 4:13, 4:13] = 17
        mask[16:26, 12:22, 11:20] = 24
        ct[mask == 17] = 500
        ct[mask == 24] = 850
        self.affine = np.diag([-1.0, 1.4, -2.1, 1.0])
        self.affine[:3, 3] = [110, -34, 47]
        self.ctpath = self.root / "ct.nii.gz"
        self.maskpath = self.root / "mask.nii.gz"
        nib.save(nib.Nifti1Image(ct, self.affine), self.ctpath)
        nib.save(nib.Nifti1Image(mask, self.affine), self.maskpath)
        self.settings = PreprocessSettings(spacing_xyz_mm=(1.2, 1.2, 1.2))

    def test_alignment_orientation_label_and_inference_consistency(self):
        img, mask, metadata = preprocess_pair(self.ctpath, self.maskpath, self.settings,
                                               expected_target_count=2)
        self.assertEqual(img.shape, mask.shape)
        self.assertEqual(img.dtype, np.float16)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 1, 8})
        self.assertEqual(metadata["processed_orientation"], ["R", "A", "S"])
        self.assertEqual(metadata["source_target_labels"], [17, 24])
        self.assertEqual(metadata["raw_shape_xyz"], [29, 25, 23])
        img_only, meta_only = preprocess_ct_only(self.ctpath, self.settings)
        np.testing.assert_array_equal(img, img_only)
        self.assertEqual(meta_only["processed_affine_xyz"], metadata["processed_affine_xyz"])

    def test_case_write_and_resume(self):
        row = {"dataset_split": "training", "patient_id": "patient5", "scan_id": "patient5_dir-ax",
               "ct_path": str(self.ctpath), "mask_path": str(self.maskpath), "target_count": "2",
               "validation_status": "passed"}
        out = self.root / "processed"
        first = write_case(row, out, self.settings, project_root=self.root)
        second = write_case(row, out, self.settings, project_root=self.root)
        self.assertEqual(first, second)
        self.assertTrue(resolve_path(first["image_path"], self.root).is_file())
        self.assertTrue(resolve_path(first["centers_path"], self.root).is_file())
        self.assertEqual(set(np.unique(np.load(resolve_path(first["mask_path"], self.root)))), {0, 1, 8})
        with self.assertRaisesRegex(ValueError, "DIFFERENT settings"):
            write_case(row, out, PreprocessSettings(spacing_xyz_mm=(3, 3, 3)), project_root=self.root)

    def test_misaligned_affine_rejected(self):
        import nibabel as nib
        mask = np.asarray(nib.load(str(self.maskpath)).dataobj)
        modified = self.affine.copy()
        modified[0, 3] += 9
        bad = self.root / "bad_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, modified), bad)
        with self.assertRaisesRegex(ValueError, "affine mismatch"):
            preprocess_pair(self.ctpath, bad, self.settings)


if __name__ == "__main__":
    unittest.main()
