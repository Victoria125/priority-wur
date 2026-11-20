"""Transparent classroom example for the BGC–metabolomics ranking algorithm.

ALL identities and observations in this file are INVENTED, for teaching only.
It does not use actual PoDP BGC labels, and must never be presented as a
biological result or an independently validated discovery.
"""
from __future__ import annotations

import json

from exp_g_bgc_feature_ranking import ranking


def classroom_data():
    genes, features = [], []
    for index in range(8):
        strain = f"FAKE_STR-{index + 1:02d}"
        x = int(index < 4)
        for bgc, found in [("FAKE_BGC_A", x), ("FAKE_BGC_B", 1-x)]:
            genes.append({
                "dataset_id": "SYNTHETIC_CLASSROOM_ONLY",
                "strain_id": strain,
                "bgc_id": bgc,
                "present": str(found),
                "source_ref": "INVENTED_FOR_EDUCATIONAL_TEST",
            })
        for feature, found in [("FAKE_SIGNAL_X", x), ("FAKE_SIGNAL_Y", 1-x)]:
            features.append({
                "dataset_id": "SYNTHETIC_CLASSROOM_ONLY",
                "strain_id": strain,
                "feature_id": feature,
                "present": str(found),
                "source_ref": "INVENTED_FOR_EDUCATIONAL_TEST",
            })
    return genes, features


def main():
    bgcs, signals = classroom_data()
    report = ranking(bgcs, signals, "SYNTHETIC_CLASSROOM_ONLY")
    print(json.dumps({
        "this_is_not_real_biology": True,
        "strains": 8,
        "meaning": "Four imaginary strains contain FAKE_BGC_A and FAKE_SIGNAL_X; the other four contain neither.",
        "all_pairs": report["ranked_candidates"],
        "no_real_links_tested": True,
    }, indent=2))


if __name__ == "__main__":
    main()
