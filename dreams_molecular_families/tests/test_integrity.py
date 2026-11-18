"""Focused regression tests for chemical annotation and spectral data integrity.

Run from repository root:
    python -m unittest discover -s dreams_molecular_families/tests -v

Only numpy, pandas, rdkit and the standard library are needed. The DreaMS
import is stubbed to exercise spectra_table() without downloading model weights.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import annotations as A  # noqa: E402
import common as C  # noqa: E402
import check_provenance as provenance  # noqa: E402


def candidates():
    """An invalid high-scoring analog must not hide a valid lower-score hit."""
    mz = Descriptors.ExactMolWt(Chem.MolFromSmiles("CCO")) + A.PROTON
    return pd.DataFrame([
        {"cluster_index": 1, "compound_name": "wrong analog",
         "smiles": "CCO", "inchi": "", "library_score": 0.99,
         "shared_peaks": 12, "mass_diff": 9.0, "library_mz": mz + 9.0,
         "adduct": "[M+H]+"},
        {"cluster_index": 1, "compound_name": "valid ethanol",
         "smiles": "CCO", "inchi": "", "library_score": 0.82,
         "shared_peaks": 8, "mass_diff": 0.001, "library_mz": mz,
         "adduct": "M+H"},
    ]), mz


class TestLibraryAnnotations(unittest.TestCase):
    def setUp(self):
        self.lib, self.mz = candidates()
        self.meta = pd.DataFrame({"cluster_index": [1, 2],
                                  "precursor_mz": [self.mz, 65.0]})

    def test_best_exact_match_survives_higher_scoring_analog(self):
        got = A.best_library_hits(self.meta, self.lib)
        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0]["compound_name"], "valid ethanol")
        self.assertEqual(got.iloc[0]["library_score"], 0.82)

    def test_no_candidate_survives_returns_empty_schema(self):
        invalid = self.lib.iloc[[0]].copy()
        got = A.best_library_hits(self.meta, invalid)
        self.assertTrue(got.empty)
        self.assertEqual(list(got.columns), list(self.lib.columns))

    def test_missing_shared_peaks_fails_closed(self):
        got = A.best_library_hits(self.meta, self.lib.drop(columns=["shared_peaks"]))
        self.assertTrue(got.empty)

    def test_missing_all_mass_evidence_fails_closed(self):
        got = A.best_library_hits(
            self.meta, self.lib.drop(columns=["mass_diff", "library_mz"]))
        self.assertTrue(got.empty)

    def test_missing_mass_diff_can_be_checked_from_both_precursors(self):
        got = A.best_library_hits(self.meta, self.lib.drop(columns=["mass_diff"]))
        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0]["compound_name"], "valid ethanol")

    def test_protonated_hit_without_reference_precursor_rejected(self):
        got = A.best_library_hits(self.meta, self.lib.drop(columns=["library_mz"]))
        self.assertTrue(got.empty)

    def test_inchi_disagreement_fails_closed(self):
        lib = self.lib.iloc[[1]].copy()
        lib["inchi"] = "InChI=1S/H2O/h1H2"
        self.assertTrue(A.best_library_hits(self.meta, lib).empty)

    def test_gnps_inchi_without_prefix_is_checked(self):
        # GNPS often stores the InChI without "InChI=": it must be read and
        # compared, not rejected as invalid (or skipped as absent).
        lib = self.lib.iloc[[1]].copy()
        lib["inchi"] = "1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        self.assertEqual(len(A.best_library_hits(self.meta, lib)), 1)
        lib["inchi"] = "1S/H2O/h1H2"
        self.assertTrue(A.best_library_hits(self.meta, lib).empty)

    def test_funnel_counts_all_candidates_not_prepared_metadata(self):
        # Spectrum 2 has only an analog hit: prepare_data drops it from the
        # metadata, but the funnel must still count it before the mass check.
        lib = pd.concat([self.lib, pd.DataFrame([{
            "cluster_index": 2, "compound_name": "analog", "smiles": "CCO", "inchi": "",
            "library_score": 0.9, "shared_peaks": 10, "mass_diff": 9.0,
            "library_mz": 74.0, "adduct": "[M+H]+"}])], ignore_index=True)
        funnel = dict(A.candidate_funnel(self.meta, lib))
        self.assertEqual(funnel["library hits with a SMILES"], 2)
        self.assertEqual(funnel[f"exact-mass hits (|query - library precursor| <= {A.MAX_MASS_DIFF} Da)"], 1)

    def test_empty_validation_keeps_expected_columns(self):
        invalid = self.lib.iloc[[0]].copy().merge(
            self.meta, on="cluster_index", how="left")
        got, counts = A.validated_annotations(invalid)
        self.assertTrue(got.empty)
        self.assertEqual(
            list(got.columns), ["row", "key", "adduct", "library_score", "mol"])
        self.assertEqual(counts[-1][1], 0)

    def test_library_reader_preserves_multiple_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "library.tsv"
            self.lib.rename(columns={
                "cluster_index": "#Scan#", "compound_name": "Compound_Name",
                "smiles": "Smiles", "inchi": "INCHI",
                "library_score": "MQScore", "shared_peaks": "SharedPeaks",
                "mass_diff": "MassDiff", "library_mz": "LibMZ",
                "adduct": "Adduct",
            }).to_csv(path, sep="\t", index=False)
            got = C.read_library_hits(path)
            self.assertEqual(len(got), 2)
            self.assertEqual(got["cluster_index"].nunique(), 1)

    def test_one_molecule_selected_after_qualification(self):
        good = self.lib.iloc[[1]].copy()
        good["precursor_mz"] = self.mz
        got, _ = A.validated_annotations(good.reset_index(drop=True))
        self.assertEqual(len(got), 1)
        self.assertEqual(int(got.iloc[0]["row"]), 0)


class TestSpectralIntegrity(unittest.TestCase):
    def test_top_peaks_sorted_even_without_truncation(self):
        spec = {"mz": np.array([50., 20., 30., 10.]),
                "intensity": np.array([1., 10., 9., 8.])}
        out = C.top_peaks(spec, k=10)
        np.testing.assert_array_equal(out["mz"], [10., 20., 30., 50.])
        np.testing.assert_array_equal(out["intensity"], [8., 10., 9., 1.])

    def test_top_peaks_sorted_after_truncation(self):
        spec = {"mz": np.array([50., 20., 30., 10.]),
                "intensity": np.array([1., 10., 9., 8.])}
        out = C.top_peaks(spec, k=2)
        np.testing.assert_array_equal(out["mz"], [20., 30.])
        np.testing.assert_array_equal(out["intensity"], [10., 9.])

    def test_embedding_cache_checks_model_revision_and_peak_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            mgf = root / "spectra.mgf"
            info = root / "embedding_info.json"
            mgf.write_text("test fixture", encoding="utf-8")
            C.write_json({"spectra_fingerprint": "fixture-fp", "model": "model-A",
                          "dreams_commit": "commit-A", "input_peaks": 100}, info)
            with mock.patch.object(C, "spectra_fingerprint", return_value="fixture-fp"):
                self.assertTrue(C.embeddings_up_to_date(
                    info, mgf, expected={"model": "model-A", "dreams_commit": "commit-A"}))
                self.assertFalse(C.embeddings_up_to_date(
                    info, mgf, expected={"model": "model-B"}))
                self.assertFalse(C.embeddings_up_to_date(
                    info, mgf, expected={"dreams_commit": "commit-B"}))
                self.assertFalse(C.embeddings_up_to_date(
                    info, mgf, expected={"input_peaks": 60}))
            with mock.patch.object(C, "spectra_fingerprint", return_value="different-fp"):
                self.assertFalse(C.embeddings_up_to_date(
                    info, mgf, expected={"model": "model-A"}))

    def test_top_peaks_rejects_bad_k_and_mismatched_lengths(self):
        spec = {"mz": np.array([2., 1.]), "intensity": np.array([2., 3.])}
        with self.assertRaises(ValueError):
            C.top_peaks(spec, k=0)
        with self.assertRaises(ValueError):
            C.top_peaks({"mz": np.array([1.]), "intensity": np.array([])})

    def test_spectral_rows_remain_aligned_after_library_selection(self):
        fake_dreams = types.ModuleType("dreams")
        fake_dreams.__path__ = []
        fake_utils = types.ModuleType("dreams.utils")
        fake_utils.__path__ = []
        fake_formats = types.ModuleType("dreams.utils.dformats")
        fake_formats.DataFormatA = type("FakeFormatA", (), {"val_spec": lambda *a, **k: True})
        with mock.patch.dict(sys.modules, {
            "dreams": fake_dreams, "dreams.utils": fake_utils,
            "dreams.utils.dformats": fake_formats,
        }):
            import prepare_data
        spectra = [
            {"scans": 1, "precursor_mz": self_mz(), "charge": 1,
             "mz": np.array([10., 20.]), "intensity": np.array([20., 30.])},
            {"scans": 2, "precursor_mz": 65., "charge": 1,
             "mz": np.array([10., 20.]), "intensity": np.array([20., 30.])},
        ]
        nodes = pd.DataFrame({"cluster_index": [1, 2], "gnps_family": [10, 10]})
        lib, _ = candidates()
        result = prepare_data.spectra_table(spectra, nodes, lib)
        self.assertEqual(len(result), len(spectra))
        self.assertEqual(result["row"].tolist(), [0, 1])
        self.assertEqual(result["cluster_index"].tolist(), [1, 2])
        self.assertEqual(result["compound_name"].iloc[0], "valid ethanol")
        self.assertTrue(pd.isna(result["compound_name"].iloc[1]))


def self_mz():
    return Descriptors.ExactMolWt(Chem.MolFromSmiles("CCO")) + A.PROTON


class TestProvenance(unittest.TestCase):
    def test_no_spectral_inputs_is_failure_not_success(self):
        with mock.patch.object(provenance, "job_input_files", return_value=[]), \
             mock.patch.object(provenance.C, "write_json"):
            result = provenance.check_run("duncan_salinispora")
        self.assertFalse(result["ok"])
        self.assertEqual(result["spectra files"], 0)
        self.assertTrue(result["problems"])


if __name__ == "__main__":
    unittest.main()
