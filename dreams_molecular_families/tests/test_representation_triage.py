"""Tests for the no-new-model-run representation triage."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from representation_triage import evaluate, table, section, ci_includes_zero, render_markdown


def demo_report():
    labels = "|  | tahlan | chevrette | missing |\n|---|---|---|---|\n"
    datasets = (
        labels
        + "| MassIVE dataset (PoDP) | MSV_T | MSV_C | MSV_M |\n"
        + "| spectra with peaks | 1000 | 100 | 20 |\n"
        + "| spectra (protocol) | 700 | 65 | 0 |\n"
        + "| spectra sources (provenance check) | ok: T | ok: C | ok: M |\n"
    )
    main = (
        labels
        + "| B independent compounds after all checks | 176 | 21 | |\n"
        + "| A1 family recovery@5: DreaMS | 0.6 | 0.7 | |\n"
        + "| A1 family recovery@5: MS2DeepScore | 0.61 | 0.65 | |\n"
        + "| A1 family recovery@5: binned cosine | 0.63 | 0.68 | |\n"
        + "| A1 DreaMS minus binned cosine (95% CI) | -0.03 [-0.06, +0.01] | +0.02 [-0.01, +0.05] | |\n"
        + "| B1 Spearman with Tanimoto: DreaMS | 0.266 | 0.265 | |\n"
        + "| B1 Spearman with Tanimoto: MS2DeepScore | 0.319 | 0.361 | |\n"
        + "| B1 Spearman with Tanimoto: modified cosine | 0.292 | 0.208 | |\n"
        + "| B1 DreaMS minus MS2DeepScore (95% CI) | -0.05 [-0.15, +0.02] | -0.10 [-0.33, +0.16] | |\n"
    )
    return (
        "# Test\n\n## Datasets\n\n" + datasets
        + "\n## Main numbers (DreaMS protocol)\n\n" + main
        + "\n## Sensitivity analysis: our cleaning (all_kept)\n\n"
        + labels + "| B independent compounds after all checks | 999 | 999 | 999 |\n"
    )


class RepresentationTriageTests(unittest.TestCase):
    def test_select_independent_chemical_reference_not_largest_genome_count(self):
        r = evaluate(demo_report(), {
            "MSV_T": {"n_genomes": 4},
            "MSV_C": {"n_genomes": 120},
        })
        self.assertEqual(r["primary_structural_benchmark_candidates"], ["tahlan"])
        runs = {v["run"]: v for v in r["runs"]}
        self.assertEqual(runs["chevrette"]["distinct_chemical_structures"], 21)
        self.assertEqual(runs["chevrette"]["chemical_benchmark_role"], "exploratory external structural set")
        self.assertEqual(runs["missing"]["protocol_spectra"], 0)
        self.assertEqual(runs["tahlan"]["in_domain_fraction"], 0.7)
        self.assertEqual(r["ready_for_bgc_spectrum_linking"], [])

    def test_sensitivity_table_is_not_silently_used(self):
        r = evaluate(demo_report(), {})
        self.assertEqual(r["runs"][0]["distinct_chemical_structures"], 176)

    def test_interval_including_zero_must_not_prove_improvement(self):
        self.assertTrue(ci_includes_zero("-0.02 [-0.03, +0.01]"))
        self.assertFalse(ci_includes_zero("+0.10 [+0.02, +0.19]"))
        self.assertTrue(ci_includes_zero("0.00 [0.00, 0.04]"))

    def test_invalid_interval_rejected(self):
        with self.assertRaisesRegex(ValueError, "bounds reversed"):
            ci_includes_zero("0.1 [+0.5, +0.1]")
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            ci_includes_zero("not a CI")

    def test_missing_main_section_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing report section"):
            evaluate(demo_report().replace("## Main numbers (DreaMS protocol)", "## Something else"), {})

    def test_duplicate_dataset_column_fails(self):
        with self.assertRaisesRegex(ValueError, "Duplicate run"):
            table("|  | tahlan | tahlan |\n|---|---|---|\n")

    def test_no_primary_reference_must_fail(self):
        with self.assertRaisesRegex(ValueError, "No dataset meets"):
            evaluate(demo_report().replace("176 | 21 |", "19 | 21 |"), {})

    def test_too_many_accepted_spectra_fails(self):
        with self.assertRaisesRegex(ValueError, "More accepted spectra"):
            evaluate(demo_report().replace("700 | 65 |", "1001 | 65 |"), {})

    def test_missing_primary_scores_fails(self):
        with self.assertRaisesRegex(ValueError, "lacks comparison scores"):
            evaluate(demo_report().replace("0.319 | 0.361 |", " | 0.361 |"), {})

    def test_report_does_not_claim_verified_links(self):
        r = evaluate(demo_report(), {})
        md = render_markdown(r)
        self.assertIn("None of the six public datasets is independently verified", md)
        self.assertIn("not random spectrum pairs", md)


if __name__ == "__main__":
    unittest.main()
