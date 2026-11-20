"""Educational example extraction uses actual data rows in aligned order."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from teaching_walkthrough import example_indices, make_examples


class WalkthroughTests(unittest.TestCase):
    def test_select_alphabetical_keys_not_favourable_models(self):
        self.assertEqual(example_indices(["Y", "A", "Z", "B"], 2), [1, 3])

    def test_duplicate_keys_cannot_be_used(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            example_indices(["A", "A"], 1)

    def test_examples_preserve_peak_values_and_neighbour(self):
        selected = pd.DataFrame({"key": ["ZZ", "AA", "MM"], "row": [0, 1, 2]})
        meta = pd.DataFrame([
            {"compound_name": "Putative_A", "cluster_index": 11, "precursor_mz": 501.,
             "gnps_family": 1},
            {"compound_name": "Putative_B", "cluster_index": 12, "precursor_mz": 502.,
             "gnps_family": 1},
            {"compound_name": "Putative_C", "cluster_index": 13, "precursor_mz": 503.,
             "gnps_family": 3},
        ])
        spectra = [
            {"mz": np.array([101., 204.]), "intensity": np.array([2., 6.])},
            {"mz": np.array([123., 256.]), "intensity": np.array([8., 4.])},
            {"mz": np.array([130., 310.]), "intensity": np.array([10., 2.])},
        ]
        s = np.array([[1., 0.9, 0.1], [0.9, 1., 0.4], [0.1, 0.4, 1.]])
        t = np.array([[1., 0.7, 0.2], [0.7, 1., 0.1], [0.2, 0.1, 1.]])
        rows = make_examples(selected, meta, spectra, {"DreaMS": s}, t, count=1)
        self.assertEqual(rows[0]["query_structure_key"], "AA")
        self.assertEqual(rows[0]["query_putative_library_name"], "Putative_B")
        self.assertEqual(rows[0]["five_strongest_real_spectral_peaks"][0]["mz"], 123.)
        self.assertEqual(rows[0]["nearest_neighbour_by_method"]["DreaMS"]["neighbour_structure_key"], "ZZ")
        self.assertTrue(rows[0]["nearest_neighbour_by_method"]["DreaMS"]["same_GNPS_molecular_family"])

    def test_alignment_errors_are_not_hidden(self):
        selected = pd.DataFrame({"key": ["ZZ", "AA", "MM"], "row": [0, 1, 2]})
        meta = pd.DataFrame([{"compound_name": "x", "cluster_index": 1, "precursor_mz": 450., "gnps_family": -1}] * 3)
        spectra = [{"mz": np.array([10., 15.]), "intensity": np.array([1.])}] * 3
        sim = np.eye(3)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            make_examples(selected, meta, spectra, {"DreaMS": sim}, np.eye(3), count=1)


if __name__ == "__main__":
    unittest.main()
