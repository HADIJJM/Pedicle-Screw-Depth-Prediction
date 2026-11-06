"""Lightweight integrity tests for Module 1 reports (no CT loading).

Run from repository root:
    python -m unittest discover -s tests/module1 -p 'test_*.py' -v
"""
from __future__ import annotations

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / 'outputs/m1_inspection/dataset_inventory.csv'
INSPECTION_SUMMARY = ROOT / 'outputs/m1_inspection/inspection_summary.json'
REPORT = ROOT / 'outputs/m1_validation/validation_report.csv'
VALIDATION_SUMMARY = ROOT / 'outputs/m1_validation/validation_summary.json'
EXPECTED_SPLITS = {'training': 61, 'validation': 80, 'test': 73}
TARGET_LABELS = set(range(17, 25))


def read_csv(path):
    with path.open('r', newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    with path.open('r', encoding='utf-8-sig') as handle:
        return json.load(handle)


def key(row):
    return (row['dataset_split'].strip().lower(), row['patient_id'].strip(), row['scan_id'].strip())


class TestModule1Reports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (INVENTORY, INSPECTION_SUMMARY, REPORT, VALIDATION_SUMMARY):
            if not path.is_file():
                raise FileNotFoundError(f'Missing required report: {path}')
        cls.inventory = read_csv(INVENTORY)
        cls.inspection = read_json(INSPECTION_SUMMARY)
        cls.report = read_csv(REPORT)
        cls.summary = read_json(VALIDATION_SUMMARY)

    def test_expected_case_counts(self):
        self.assertEqual(len(self.inventory), 214)
        self.assertEqual(len(self.report), 214)
        for rows in (self.inventory, self.report):
            self.assertEqual(dict(Counter(r['dataset_split'].strip().lower() for r in rows)), EXPECTED_SPLITS)

    def test_unique_cases(self):
        for rows in (self.inventory, self.report):
            keys = [key(r) for r in rows]
            self.assertEqual(len(keys), len(set(keys)), 'Duplicate scan key detected')

    def test_no_patient_overlap(self):
        for rows in (self.inventory, self.report):
            owner = {}
            for r in rows:
                patient, split = r['patient_id'].strip(), r['dataset_split'].strip().lower()
                if patient in owner:
                    self.assertEqual(owner[patient], split, f'Patient {patient} crosses splits')
                owner[patient] = split

    def test_same_cases_in_inventory_and_report(self):
        self.assertEqual({key(r) for r in self.inventory}, {key(r) for r in self.report})

    def test_inspection_summary(self):
        self.assertEqual(int(self.inspection['scans']), len(self.inventory))
        self.assertEqual(self.inspection['scans_per_split'], { 'training': 61, 'validation': 80, 'test': 73 })
        coverage = Counter(r['coverage_type'].strip().lower() for r in self.inventory)
        self.assertEqual(dict(coverage), self.inspection['coverage_type'])
        self.assertEqual(sum(coverage.values()), 214)

    def test_validation_summary(self):
        statuses = Counter(r['validation_status'].strip().lower() for r in self.report)
        self.assertEqual(int(self.summary['total_scans']), len(self.report))
        self.assertEqual(int(self.summary['expected_scans']), len(self.inventory))
        self.assertEqual(int(self.summary['remaining']), 0)
        self.assertEqual(int(self.summary['passed']), statuses.get('passed', 0))
        self.assertEqual(int(self.summary['warnings']), statuses.get('warning', 0))
        self.assertEqual(int(self.summary['failed']), statuses.get('failed', 0))
        self.assertEqual(dict(statuses), self.summary['status_counts'])
        self.assertEqual(statuses.get('failed', 0), 0)
        self.assertEqual(set(statuses), {'passed', 'warning'})

    def test_no_target_warnings_match_inspection(self):
        inv = {key(r): r for r in self.inventory}
        for r in self.report:
            k = key(r)
            target_count = int(float(r['target_count']))
            original_count = int(float(inv[k]['target_count']))
            self.assertEqual(target_count, original_count, f'Target count mismatch: {k}')
            self.assertEqual(r['coverage_type'].strip().lower(), inv[k]['coverage_type'].strip().lower())
            no_target = target_count == 0
            self.assertEqual(no_target, 'no_target_vertebrae' in r['issues'], f'Warning mismatch: {k}')
            if no_target:
                self.assertEqual(r['validation_status'], 'warning')
                self.assertEqual(inv[k]['coverage_type'].strip().lower(), 'none')
            else:
                self.assertEqual(r['validation_status'], 'passed')

    def test_passed_cases_have_matching_metadata(self):
        for r in self.report:
            if r['validation_status'] != 'passed':
                continue
            for field in ('shape_match', 'spacing_match', 'affine_match', 'orientation_match',
                          'mask_integer_labels', 'mask_finite_labels'):
                self.assertEqual(r[field].strip().lower(), 'true', f'{field} false for {key(r)}')
            self.assertFalse(r['unexpected_labels'].strip(), f'Unexpected labels: {key(r)}')
            self.assertEqual(r['centroid_status'].strip(), 'readable')
            self.assertFalse(r['issues'].strip(), f'Unexpected issues: {key(r)}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
