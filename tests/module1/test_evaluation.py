"""Evaluation tests: metrics, label mapping, grids, test-split protocol (synthetic data only).

Run from repository root:
    python -m unittest discover -s tests/module1 -p "test_evaluation.py" -v
"""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.module1_segmentation.evaluation import metrics as M

NIBABEL = importlib.util.find_spec("nibabel") is not None and importlib.util.find_spec("scipy") is not None
TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("monai") is not None
MPL = importlib.util.find_spec("matplotlib") is not None


def stacked(levels=range(1, 9), shape=(20, 20, 90), step=10):
    vol = np.zeros(shape, dtype=np.uint8)
    for c in levels:
        vol[5:15, 5:15, c * step:c * step + step - 2] = c
    return vol


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        gt = stacked()
        r = M.evaluate_volumes(gt, gt.copy(), np.eye(4))
        self.assertEqual(r["case"]["mean_dice"], 1.0)
        self.assertEqual(r["case"]["failure_category"], "ok")
        self.assertEqual(r["case"]["id_rate"], 1.0)

    def test_nan_and_zero_rules_match_training(self):
        gt = stacked(levels=[3, 4, 5])
        pred = stacked(levels=[1, 3, 4])           # T10 false positive, L2 missed, T11 absent in both
        dice, iou = M.dice_iou_from_confusion(M.confusion_matrix(gt, pred))
        self.assertEqual(dice[0], 0.0)              # T10 predicted only
        self.assertTrue(np.isnan(dice[1]))          # T11 absent in both
        self.assertEqual(dice[2], 1.0)
        self.assertEqual(dice[4], 0.0)              # L2 reference only
        self.assertTrue(np.isnan(iou[1]))
        r = M.evaluate_volumes(gt, pred, np.eye(4))
        out = {lv["level"]: lv["outcome"] for lv in r["levels"]}
        self.assertEqual((out["T10"], out["T11"], out["L2"]), ("false_positive", "absent", "missed"))

    @unittest.skipUnless(TORCH, "torch/monai not installed")
    def test_identical_to_training_dice(self):
        from src.module1_segmentation.training.train import dice_per_class
        rng = np.random.default_rng(0)
        for _ in range(5):
            gt = rng.integers(0, 9, size=(12, 13, 14)).astype(np.uint8)
            pred = rng.integers(0, 9, size=(12, 13, 14)).astype(np.uint8)
            gt[gt == rng.integers(1, 9)] = 0  # make some classes absent
            mine, _ = M.dice_iou_from_confusion(M.confusion_matrix(gt, pred))
            np.testing.assert_allclose(mine, dice_per_class(pred, gt), equal_nan=True)

    def test_iou_dice_relation_and_chunking(self):
        rng = np.random.default_rng(1)
        gt = rng.integers(0, 9, size=(30, 31, 32)).astype(np.uint8)
        pred = rng.integers(0, 9, size=(30, 31, 32)).astype(np.uint8)
        full = M.confusion_matrix(gt, pred)
        with mock.patch.object(M, "_CHUNK_VOXELS", 997):
            np.testing.assert_array_equal(M.confusion_matrix(gt, pred), full)
        self.assertEqual(int(full.sum()), gt.size)
        dice, iou = M.dice_iou_from_confusion(full)
        np.testing.assert_allclose(iou, dice / (2 - dice))

    def test_level_shift_detected_both_directions(self):
        gt = stacked()
        too_cranial = np.where(gt >= 2, gt - 1, 0).astype(np.uint8)   # true L3 called "L2"
        too_caudal = np.where((gt >= 1) & (gt <= 7), gt + 1, 0).astype(np.uint8)
        a = M.evaluate_volumes(gt, too_cranial, np.eye(4))["case"]
        b = M.evaluate_volumes(gt, too_caudal, np.eye(4))["case"]
        self.assertEqual((a["best_shift"], a["failure_category"]), (1, "level_shift"))
        self.assertEqual((b["best_shift"], b["failure_category"]), (-1, "level_shift"))
        self.assertAlmostEqual(a["mean_dice_at_best_shift"], 7 / 8)
        self.assertGreater(a["spine_dice"], 0.9)
        self.assertEqual(a["mean_dice"], 0.0)

    def test_identification_needs_nearest_reference(self):
        gt = stacked()
        r = M.evaluate_volumes(gt, np.where(gt >= 2, gt - 1, 0).astype(np.uint8), np.eye(4))
        t11 = r["levels"][1]
        self.assertLess(t11["centroid_distance_mm"], M.ID_DISTANCE_MM)  # 10 mm away...
        self.assertEqual(t11["identified"], 0)                          # ...but sits on another level

    def test_fragments_and_mm_volumes(self):
        gt = stacked(levels=[4])
        pred = gt.copy()
        pred[0:2, 0:2, 0:2] = 4  # stray island
        aff = np.diag([2.0, 2.0, 2.0, 1.0])
        lv = M.evaluate_volumes(gt, pred, aff)["levels"][3]
        self.assertEqual(lv["pred_components"], 2)
        self.assertAlmostEqual(lv["gt_volume_ml"], 10 * 10 * 8 * 8 / 1000)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            M.confusion_matrix(np.zeros((2, 2, 2), np.uint8), np.zeros((2, 2, 3), np.uint8))
        with self.assertRaises(ValueError):
            M.confusion_matrix(np.full((2, 2, 2), 17, np.uint8), np.zeros((2, 2, 2), np.uint8))
        with self.assertRaises(TypeError):
            M.confusion_matrix(np.zeros((2, 2, 2), np.float32), np.zeros((2, 2, 2), np.uint8))


@unittest.skipUnless(NIBABEL, "nibabel/scipy not installed")
class TestCaseIO(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        from src.module1_segmentation.evaluation import case_io
        self.nib, self.io = nib, case_io
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, arr, aff):
        p = self.root / name
        self.nib.save(self.nib.Nifti1Image(arr, aff), str(p))
        return p

    def test_verse_mapping_matches_preprocessing(self):
        src = np.array([[[0, 1, 16, 17, 18, 24, 25, 28]]], dtype=np.uint8)
        self.assertEqual(self.io.map_verse_labels(src).tolist(), [[[0, 0, 0, 1, 2, 8, 0, 0]]])
        with self.assertRaises(ValueError):
            self.io.map_verse_labels(np.array([[[30]]], dtype=np.uint8))

    def test_prediction_with_verse_labels_gets_a_hint(self):
        p = self.save("p.nii.gz", np.full((4, 4, 4), 20, np.uint8), np.eye(4))
        with self.assertRaisesRegex(ValueError, "--pred-labels verse"):
            self.io.load_prediction(p)
        self.assertEqual(int(self.io.load_prediction(p, "verse")[0].max()), 4)

    def test_float_reference_and_off_grid_prediction(self):
        gt = np.zeros((10, 10, 10), np.float32)
        gt[2:8, 2:8, 2:8] = 20.0
        g = self.save("g.nii.gz", gt, np.eye(4))
        shifted = np.eye(4)
        shifted[0, 3] = 0.5
        p = self.save("p.nii.gz", (gt > 0).astype(np.uint8) * 4, shifted)
        vols = self.io.load_case(g, p)
        self.assertTrue(vols.grid_resampled)
        self.assertEqual(vols.pred.shape, vols.gt.shape)
        with self.assertRaisesRegex(ValueError, "original CT grid"):
            self.io.load_case(g, p, allow_resample=False)
        bad = gt.copy()
        bad[0, 0, 0] = 17.5
        with self.assertRaisesRegex(ValueError, "non-integer"):
            self.io.load_reference(self.save("bad.nii.gz", bad, np.eye(4)))


@unittest.skipUnless(NIBABEL, "nibabel/scipy not installed")
class TestEvaluateSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.module1_segmentation.evaluation.synthetic import make_synthetic_split
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.manifest, cls.pred_dir = make_synthetic_split(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_eval(self, **kw):
        from src.module1_segmentation.evaluation.evaluate import EvalConfig, evaluate_split
        cfg = EvalConfig(pred_dir=str(kw.pop("pred_dir", self.pred_dir)), manifest=str(kw.pop("manifest", self.manifest)),
                         output_root=str(self.root / "evaluation_out"), tag=kw.pop("tag", "t"), figures=kw.pop("figures", "none"), **kw)
        return evaluate_split(cfg, project_root=self.root, log=lambda *_: None)

    def test_categories_and_outputs(self):
        res = self.run_eval()
        out = Path(res["output_dir"])
        for name in ("per_case.csv", "per_level.csv", "summary_per_level.csv", "failure_cases.csv",
                     "confusion_voxels.csv", "level_assignment.csv", "summary.json", "report.md"):
            self.assertTrue((out / name).is_file(), name)
        with (out / "per_case.csv").open(encoding="utf-8") as f:
            cases = {r["scan_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(cases["syn-perfect"]["failure_category"], "ok")
        self.assertEqual(cases["syn-shift"]["failure_category"], "level_shift")
        self.assertEqual(cases["syn-shift"]["best_shift"], "-1")
        self.assertEqual(cases["syn-missfp"]["failure_category"], "missed_levels")
        self.assertEqual(cases["syn-missfp"]["n_false_positive"], "1")
        self.assertGreater(float(cases["syn-pipeline"]["mean_dice"]), 0.9)  # preprocessing -> inference save round trip (LPS)
        self.assertEqual(cases["syn-offgrid"]["grid_resampled"], "1")
        self.assertGreater(float(cases["syn-offgrid"]["mean_dice"]), 0.9)
        self.assertEqual(res["n_cases"], 5)
        # mean over levels == training-time definition (mean of per-level means)
        lv = [r["dice_mean"] for r in res["per_level"]]
        self.assertAlmostEqual(res["overall"]["mean_dice_levels"], float(np.nanmean(lv)))

    def test_resume_uses_cache(self):
        self.run_eval(tag="resume")
        with mock.patch("src.module1_segmentation.evaluation.evaluate.load_case",
                        side_effect=AssertionError("should be cached")):
            self.run_eval(tag="resume")

    def test_missing_predictions(self):
        empty = self.root / "empty_preds"
        empty.mkdir(exist_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, "predictions missing"):
            self.run_eval(pred_dir=empty, tag="missing")

    @unittest.skipUnless(MPL, "matplotlib not installed")
    def test_figures(self):
        res = self.run_eval(tag="figs", figures="failures")
        figs = Path(res["output_dir"]) / "figures"
        self.assertTrue((figs / "dice_per_level.png").is_file())
        self.assertTrue((figs / "level_assignment.png").is_file())
        self.assertTrue((figs / "cases" / "syn-shift.png").is_file())
        self.assertFalse((figs / "cases" / "syn-perfect.png").exists())

    def test_test_split_protocol(self):
        test_manifest = self.root / "test.csv"
        with self.manifest.open(encoding="utf-8-sig") as f:
            rows = [dict(r, dataset_split="test") for r in csv.DictReader(f)]
        with test_manifest.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with self.assertRaises(PermissionError):
            self.run_eval(split="test", manifest=test_manifest, tag="final")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_eval(split="test", manifest=test_manifest, tag="final", final_test=True)
        prov = self.pred_dir / "predictions_manifest.json"
        try:
            prov.write_text(json.dumps({"dataset_split": "test", "checkpoint_sha256": "a" * 64}))
            self.run_eval(split="test", manifest=test_manifest, tag="final", final_test=True)
            lock = json.loads((self.root / "outputs/evaluation/test_lock.json").read_text())
            self.assertEqual(lock["checkpoint_sha256"], "a" * 64)
            self.run_eval(split="test", manifest=test_manifest, tag="final", final_test=True)  # same model: ok
            prov.write_text(json.dumps({"dataset_split": "test", "checkpoint_sha256": "b" * 64}))
            with self.assertRaisesRegex(PermissionError, "already evaluated"):
                self.run_eval(split="test", manifest=test_manifest, tag="final2", final_test=True)
            with self.assertRaisesRegex(ValueError, "not 'validation'"):
                self.run_eval(tag="wrong_split")  # test predictions used as validation
        finally:
            prov.unlink(missing_ok=True)


@unittest.skipUnless(NIBABEL, "nibabel not installed")
class TestOrientation(unittest.TestCase):
    def test_ras_and_lps_display_identically(self):
        from src.module1_segmentation.evaluation.synthetic import _affine, _phantom, _to_lps
        from src.module1_segmentation.evaluation.visualize import to_ras
        _, mask = _phantom([17, 18, 19])
        zooms = (1.2, 1.2, 2.5)
        a, za = to_ras(mask, _affine(), zooms)
        b, zb = to_ras(_to_lps(mask), _affine(lps=True), zooms)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(za, zb)


if __name__ == "__main__":
    unittest.main()
