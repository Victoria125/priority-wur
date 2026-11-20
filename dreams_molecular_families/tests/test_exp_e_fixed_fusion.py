"""Scientific guardrails around the fixed no-training fusion baseline."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exp_e_fixed_fusion import cosine_matrix, fusion_scores, evaluate


class FixedFusionTests(unittest.TestCase):
    def test_identical_input_scores_are_unchanged(self):
        x = np.eye(3)
        scores = fusion_scores(x, x, alpha=0.5)
        np.testing.assert_allclose(scores["DreaMS"], scores["fixed_50_50_fusion"])
        np.testing.assert_allclose(scores["MS2DeepScore"], scores["fixed_50_50_fusion"])

    def test_weighted_score_fusion_not_vector_dimension_fusion(self):
        x = np.eye(3)
        y = np.array([[1., 0.], [0., 1.], [1., 1.]])
        scores = fusion_scores(x, y)
        np.testing.assert_allclose(scores["fixed_50_50_fusion"],
                                   0.5 * scores["DreaMS"] + 0.5 * scores["MS2DeepScore"])
        self.assertEqual(scores["fixed_50_50_fusion"].shape, (3, 3))

    def test_reject_mismatch(self):
        with self.assertRaisesRegex(ValueError, "SAME spectra"):
            fusion_scores(np.eye(3), np.eye(4))

    def test_reject_zero_and_nan(self):
        with self.assertRaisesRegex(ValueError, "Zero-length"):
            cosine_matrix(np.array([[0., 0.], [1., 0.], [0., 1.]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            cosine_matrix(np.array([[np.nan, 0.], [1., 0.], [0., 1.]]))

    def test_reject_tuned_weight_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            fusion_scores(np.eye(3), np.eye(3), alpha=2)

    def test_evaluate_20_unique_compounds(self):
        x = np.eye(20)
        t = np.tile(np.linspace(0, 1, 20), (20, 1))
        t = (t + t.T) / 2
        np.fill_diagonal(t, 1.0)
        metrics = evaluate({"demo": cosine_matrix(x)}, t, [f"chem-{i}" for i in range(20)])
        self.assertIn("spearman_with_tanimoto", metrics["demo"])

    def test_no_duplicate_molecule_or_constant_target(self):
        x = np.eye(20)
        t = np.eye(20)
        keys = [f"mol-{i}" for i in range(20)]
        with self.assertRaisesRegex(ValueError, "Duplicate molecule"):
            evaluate({"demo": cosine_matrix(x)}, t, keys[:19] + [keys[0]])
        with self.assertRaisesRegex(ValueError, "constant"):
            evaluate({"demo": cosine_matrix(x)}, np.ones((20, 20)), keys)

    def test_asymmetry_rejected(self):
        n = 20
        mat = np.eye(n)
        mat[0, 1] = 0.7
        t = np.eye(n)
        t[0, 1] = t[1, 0] = 0.2
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            evaluate({"demo": mat}, t, [f"mol-{i}" for i in range(n)])


if __name__ == "__main__":
    unittest.main()
