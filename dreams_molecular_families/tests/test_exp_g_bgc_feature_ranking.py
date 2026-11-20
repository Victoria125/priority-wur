"""Synthetic-data regression tests for source-backed BGC↔metabolite ranking.

All strain associations here are intentionally fictional. These tests verify
algorithmic invariants only, not real biological link accuracy.
"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from exp_g_bgc_feature_ranking import (
    benjamini_hochberg,
    evaluate_known_positives,
    phi_coefficient,
    presence_matrix,
    ranking,
    read_csv,
)


def fake_data(n=8):
    genes = []
    features = []
    for i in range(n):
        strain = f"S{i}"
        x = int(i < (n // 2))
        for bgc, present in (("BGC_A", x), ("BGC_B", 1-x), ("BGC_C", x)):
            genes.append(dict(dataset_id="SYNTHETIC", strain_id=strain,
                              bgc_id=bgc, present=str(present),
                              source_ref="SYNTHETIC_TEST_ONLY"))
        for feature, present in (("FEATURE_X", x), ("FEATURE_Y", 1-x)):
            features.append(dict(dataset_id="SYNTHETIC", strain_id=strain,
                                 feature_id=feature, present=str(present),
                                 source_ref="SYNTHETIC_TEST_ONLY"))
    return genes, features


class IndependentLinkRankingTests(unittest.TestCase):
    def test_known_synthetic_cooccurrence_ranked_first(self):
        genes, features = fake_data()
        result = ranking(genes, features, "SYNTHETIC")
        self.assertEqual(result["curated_strains"], 8)
        self.assertEqual(result["bgc_count"], 3)
        self.assertEqual(result["metabolomic_feature_count"], 2)
        by_pair = {(x["bgc_id"], x["feature_id"]): x for x in result["ranked_candidates"]}
        self.assertEqual(by_pair["BGC_A", "FEATURE_X"]["rank_within_bgc"], 1)
        self.assertEqual(by_pair["BGC_B", "FEATURE_Y"]["rank_within_bgc"], 1)
        self.assertEqual(by_pair["BGC_A", "FEATURE_X"]["n11"], 4)
        self.assertEqual(by_pair["BGC_A", "FEATURE_X"]["n00"], 4)
        self.assertGreater(by_pair["BGC_A", "FEATURE_X"]["phi"], 0)
        self.assertLess(by_pair["BGC_A", "FEATURE_X"]["fisher_one_sided_p"], 0.05)
        self.assertTrue(all(0 <= x["bh_fdr_q"] <= 1 for x in result["ranked_candidates"]))
        self.assertTrue(all(x["label"].startswith("HYPOTHESIS") for x in result["ranked_candidates"]))

    def test_incomplete_presence_is_not_absence(self):
        genes, features = fake_data()
        features.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete feature matrix"):
            ranking(genes, features, "SYNTHETIC")

    def test_unknown_observation_rejected(self):
        genes, features = fake_data()
        genes[0]["present"] = "unknown"
        with self.assertRaisesRegex(ValueError, "explicit 0 or 1"):
            ranking(genes, features, "SYNTHETIC")

    def test_duplicate_cell_rejected_even_if_same_value(self):
        genes, features = fake_data()
        with self.assertRaisesRegex(ValueError, "Duplicate observation"):
            ranking(genes + [dict(genes[0])], features, "SYNTHETIC")

    def test_modalities_have_same_strains(self):
        genes, features = fake_data()
        for row in features:
            if row["strain_id"] == "S7":
                row["strain_id"] = "S9"
        with self.assertRaisesRegex(ValueError, "EXACTLY the same curated strains"):
            ranking(genes, features, "SYNTHETIC")

    def test_constant_genomic_variable_rejected(self):
        genes, features = fake_data()
        for row in genes:
            if row["bgc_id"] == "BGC_A":
                row["present"] = "1"
        with self.assertRaisesRegex(ValueError, "Uninformative BGC"):
            ranking(genes, features, "SYNTHETIC")

    def test_too_few_strains_rejected(self):
        genes, _ = fake_data(n=7)
        with self.assertRaisesRegex(ValueError, "minimum 8"):
            presence_matrix(genes, "SYNTHETIC", "bgc")

    def test_fdr_properties(self):
        inp = [0.01, 0.03, 0.06, 0.10]
        q = benjamini_hochberg(inp)
        self.assertTrue(all(v >= p for v, p in zip(q, inp)))
        self.assertEqual(sorted(q), q)
        with self.assertRaisesRegex(ValueError, "Invalid P values"):
            benjamini_hochberg([float("nan")])

    def test_signed_phi_sanity(self):
        self.assertAlmostEqual(phi_coefficient(4, 0, 0, 4), 1)
        self.assertAlmostEqual(phi_coefficient(0, 4, 4, 0), -1)
        self.assertAlmostEqual(phi_coefficient(2, 2, 2, 2), 0)

    def test_verified_positives_without_enough_labels_not_evaluated(self):
        genes, features = fake_data()
        result = ranking(genes, features, "SYNTHETIC")
        one = [{"dataset_id": "SYNTHETIC", "bgc_id": "BGC_A",
                "feature_id": "FEATURE_X", "evidence_type": "experimentally_verified",
                "evidence_ref": "SYNTHETIC_TEST_ONLY"}]
        report = evaluate_known_positives(result, one)
        self.assertIsNone(report["metrics"])
        self.assertEqual(report["verified_positives"], 1)

    def test_three_synthetic_positives_metrics_are_descriptive(self):
        genes, features = fake_data()
        result = ranking(genes, features, "SYNTHETIC")
        labels = [
            {"dataset_id": "SYNTHETIC", "bgc_id": b, "feature_id": f,
             "evidence_type": "experimentally_verified",
             "evidence_ref": "SYNTHETIC_TEST_ONLY"}
            for b, f in [("BGC_A", "FEATURE_X"), ("BGC_B", "FEATURE_Y"), ("BGC_C", "FEATURE_X")]
        ]
        report = evaluate_known_positives(result, labels)
        self.assertEqual(report["metrics"]["recall_at_1"], 1.0)
        self.assertEqual(report["metrics"]["mean_reciprocal_rank"], 1.0)
        self.assertIn("DESCRIPTIVE", report["status"])

    def test_claimed_positive_not_experimental_is_not_ground_truth(self):
        genes, features = fake_data()
        result = ranking(genes, features, "SYNTHETIC")
        labels = [{
            "dataset_id": "SYNTHETIC", "bgc_id": "BGC_A",
            "feature_id": "FEATURE_X", "evidence_type": "reported",
            "evidence_ref": "SYNTHETIC_TEST_ONLY",
        }]
        self.assertEqual(evaluate_known_positives(result, labels)["verified_positives"], 0)

    def test_verified_label_missing_from_universe_rejected(self):
        genes, features = fake_data()
        result = ranking(genes, features, "SYNTHETIC")
        labels = [
            {"dataset_id": "SYNTHETIC", "bgc_id": b, "feature_id": f,
             "evidence_type": "experimentally_verified",
             "evidence_ref": "SYNTHETIC_TEST_ONLY"}
            for b, f in [("BGC_A", "FEATURE_X"), ("BGC_B", "FEATURE_Y"), ("BGC_C", "MISSING")]
        ]
        with self.assertRaisesRegex(ValueError, "outside ranked candidate"):
            evaluate_known_positives(result, labels)

    def test_csv_rejects_blank_source_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bgcs.csv"
            p.write_text("dataset_id,strain_id,bgc_id,present,source_ref\nSYNTHETIC,S1,A,1,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank required"):
                read_csv(p, ("dataset_id", "strain_id", "bgc_id", "present", "source_ref"))

    def test_real_dataset_cli_blocks_with_missing_curated_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blocked.json"
            done = subprocess.run(
                [sys.executable, str(SCRIPTS / "exp_g_bgc_feature_ranking.py"),
                 "--dataset", "MSV000078836", "--output", str(path)],
                text=True, capture_output=True,
            )
            self.assertEqual(done.returncode, 2)
            j = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(j["no_ranking_generated"])
            self.assertIn("BLOCKED", j["status"])


if __name__ == "__main__":
    unittest.main()
