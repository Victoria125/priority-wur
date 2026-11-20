"""Tests for external-data evaluation: fail closed on leakage and small n."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp_f_external_validation import disjoint_external_keys, cross_dataset_report


class ExternalValidationTests(unittest.TestCase):
    def data(self, n=21):
        rng = np.random.default_rng(12)
        dreams = rng.normal(size=(n, 11))
        ms2ds = rng.normal(size=(n, 9))
        target = rng.uniform(0, 1, (n, n))
        target = (target + target.T) / 2
        np.fill_diagonal(target, 1)
        return dreams, ms2ds, target

    def test_drop_overlapping_connectivity_before_scoring(self):
        idx, overlap = disjoint_external_keys(["A", "B"], ["C", "A", "D"])
        self.assertEqual(list(idx), [0, 2])
        self.assertEqual(overlap, ["A"])

    def test_abort_when_overlap_removes_enough_to_fall_below_20(self):
        keys = [f"C{i}" for i in range(21)]
        d, m, t = self.data()
        r = cross_dataset_report(keys[:2], keys, d, m, t)
        self.assertEqual(r["external_compounds_after_overlap_filter"], 19)
        self.assertFalse(r["sufficient_for_exploratory_evaluation"])
        self.assertIsNone(r["scores"])

    def test_20_disjoint_compounds_are_scored(self):
        keys = [f"C{i}" for i in range(21)]
        d, m, t = self.data()
        r = cross_dataset_report(["C0"], keys, d, m, t)
        self.assertEqual(r["external_compounds_after_overlap_filter"], 20)
        self.assertTrue(r["sufficient_for_exploratory_evaluation"])
        self.assertIn("fixed_50_50_fusion", r["scores"])
        self.assertIn("fusion_minus_MS2DeepScore", r["paired_exploratory_uncertainty"])

    def test_no_overlap_does_not_drop_data(self):
        keys = [f"C{i}" for i in range(21)]
        d, m, t = self.data()
        r = cross_dataset_report(["Z0"], keys, d, m, t)
        self.assertEqual(r["external_compounds_after_overlap_filter"], 21)
        self.assertEqual(r["overlap_with_reference_count"], 0)

    def test_reject_repeated_connectivity(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            disjoint_external_keys(["X"], ["A", "A"])

    def test_reject_misaligned_model_rows(self):
        keys = [f"C{i}" for i in range(21)]
        d, m, t = self.data()
        with self.assertRaisesRegex(ValueError, "align"):
            cross_dataset_report([], keys, d[:-1], m, t)


if __name__ == "__main__":
    unittest.main()
