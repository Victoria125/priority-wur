"""Regression tests for conservative genome–metabolome evidence checks.

No network, GPU, published model weights, or biological truth assumptions.
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from linking_evidence import audit, load_rows  # noqa: E402


PODP = [{"metabolomics_id": "MSV1", "mibig": "BGC001", "bgc_type": "MIBiG number associated with this exact BGC", "link_type": "GNPS molecular family"}]
SPECTRA = [
    {"dataset_id": "MSV1", "spectrum_id": "scan:1", "strain_id": "strain-A", "source_ref": "publication:A"},
    {"dataset_id": "MSV1", "spectrum_id": "scan:2", "strain_id": "strain-B", "source_ref": "publication:B"},
]
BGCS = [
    {"dataset_id": "MSV1", "strain_id": "strain-A", "bgc_id": "BGC001", "source_ref": "genome:A"},
    {"dataset_id": "MSV1", "strain_id": "strain-B", "bgc_id": "BGC002", "source_ref": "genome:B"},
]
VERIFIED = [
    {"dataset_id": "MSV1", "spectrum_id": "scan:1", "bgc_id": "BGC001",
     "evidence_type": "experimentally_verified", "evidence_ref": "publication:A:figure2"},
]


class EvidenceGateTests(unittest.TestCase):
    def test_podp_rows_alone_never_mean_ready(self):
        r = audit(PODP, "MSV1")
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertEqual(r["internally_consistent_verified_pairs"], 0)
        self.assertTrue(any("Missing curated" in x for x in r["issues"]))

    def test_curated_complete_with_distinct_strain_profiles_is_review_candidate(self):
        r = audit(PODP, "MSV1", SPECTRA, BGCS, VERIFIED)
        self.assertTrue(r["ready_for_manual_benchmark_review"], r["issues"])
        self.assertEqual(r["internally_consistent_verified_pairs"], 1)
        self.assertEqual(r["mapped_strains_with_both_modalities"], 2)
        self.assertIn("Not biological validation", r["caveat"])

    def test_reported_link_is_not_experimental_positive(self):
        reported = [dict(VERIFIED[0], evidence_type="reported")]
        r = audit(PODP, "MSV1", SPECTRA, BGCS, reported)
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertEqual(r["internally_consistent_verified_pairs"], 0)
        self.assertEqual(r["independently_curated_evidence_types"]["reported"], 1)

    def test_wrong_strain_bgc_join_is_not_positive(self):
        wrong = [dict(VERIFIED[0], bgc_id="BGC002")]
        r = audit(PODP, "MSV1", SPECTRA, BGCS, wrong)
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertEqual(r["internally_consistent_verified_pairs"], 0)
        self.assertTrue(any("lacks corresponding BGC" in x for x in r["issues"]))

    def test_conflicting_strain_assignment_fails_closed(self):
        bad = SPECTRA + [dict(SPECTRA[0], strain_id="strain-B")]
        r = audit(PODP, "MSV1", bad, BGCS, VERIFIED)
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertTrue(any("Conflicting" in x for x in r["issues"]))

    def test_one_strain_is_not_a_correlation_benchmark(self):
        r = audit(PODP, "MSV1", SPECTRA[:1], BGCS[:1], VERIFIED)
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertTrue(any("Fewer than two strains" in x for x in r["issues"]))

    def test_same_bgc_profiles_do_not_support_strain_variation(self):
        same = BGCS + [dict(BGCS[0], strain_id="strain-B")]
        r = audit(PODP, "MSV1", SPECTRA, same, VERIFIED)
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertTrue(any("No observable BGC-presence variation" in x for x in r["issues"]))

    def test_dataset_isolation_and_unlisted_dataset(self):
        other = [dict(SPECTRA[0], dataset_id="MSV2")]
        r = audit(PODP, "MSV2", other, [], [])
        self.assertEqual(r["podp_reported_rows"], 0)
        self.assertEqual(r["mapped_spectra"], 1)
        self.assertFalse(r["ready_for_manual_benchmark_review"])

    def test_missing_columns_and_blank_ids_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "evidence.csv"
            p.write_text("dataset_id,spectrum_id\nMSV1,scan:1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_rows(p, ("dataset_id", "spectrum_id", "strain_id"))
            p.write_text("dataset_id,spectrum_id,strain_id\nMSV1,scan:1,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blank required"):
                load_rows(p, ("dataset_id", "spectrum_id", "strain_id"))

    def test_no_truth_even_with_maps_fails_closed(self):
        r = audit(PODP, "MSV1", SPECTRA, BGCS, [])
        self.assertFalse(r["ready_for_manual_benchmark_review"])
        self.assertTrue(any("No BGC-to-spectrum evidence rows" in x for x in r["issues"]))


if __name__ == "__main__":
    unittest.main()
