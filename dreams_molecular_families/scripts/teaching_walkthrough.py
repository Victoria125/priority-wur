"""A readable, reproducible walkthrough on *real cached* Tahlan spectra.

Educational inspection, never a new training run or independent validation.
To avoid selecting success stories, take the first three independent chemical
connectivity keys in alphabetical order. Show true stored m/z peaks, the real
library annotation (putative), each model's top-1 neighbour, cosine similarity,
and chemical similarity from library-annotated structures (proxy only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def example_indices(keys: list[str], n: int = 3) -> list[int]:
    if len(keys) != len(set(keys)):
        raise ValueError("Independent structures must have distinct connectivity keys")
    if n < 1 or n > len(keys):
        raise ValueError("Requested number of examples out of range")
    return sorted(range(len(keys)), key=lambda i: keys[i])[:n]


def make_examples(selected, meta, spectra, scores, tanimoto, count=3) -> list[dict]:
    indices = example_indices(selected["key"].tolist(), count)
    n = len(selected)
    if n < 2:
        raise ValueError("Need multiple annotated compounds")
    results = []
    for q in indices:
        row = selected.iloc[q]
        meta_index = int(row["row"])
        recorded = meta.iloc[meta_index]
        spec = spectra[meta_index]
        peaks = np.asarray(spec["mz"], dtype=float)
        intens = np.asarray(spec["intensity"], dtype=float)
        if peaks.size == 0 or intens.shape != peaks.shape:
            raise ValueError("Peak intensities and m/z values are missing or misaligned")
        strongest = np.argsort(-intens, kind="stable")[:5]
        top_peaks = [
            {"mz": round(float(peaks[i]), 4),
             "relative_intensity": round(float(intens[i] / intens.max()), 4)}
            for i in strongest
        ]
        matches = {}
        for method, sim in scores.items():
            values = np.asarray(sim[q], dtype=float).copy()
            if len(values) != n:
                raise ValueError("Similarity columns are misaligned")
            values[q] = -np.inf
            neighbour = int(np.argmax(values))
            other = meta.iloc[int(selected.iloc[neighbour]["row"])]
            matches[method] = {
                "neighbour_putative_library_name": str(other.get("compound_name") or "").strip(),
                "neighbour_structure_key": str(selected.iloc[neighbour]["key"]),
                "cosine_similarity_or_weighted_score": round(float(values[neighbour]), 4),
                "morgan_tanimoto_of_putative_structures": round(float(tanimoto[q, neighbour]), 4),
                "same_GNPS_molecular_family": bool(
                    recorded.get("gnps_family", -1) != -1
                    and recorded.get("gnps_family") == other.get("gnps_family")
                ),
            }
        results.append({
            "query_putative_library_name": str(recorded.get("compound_name") or "").strip(),
            "query_structure_key": str(row["key"]),
            "query_scan_id": str(recorded.get("cluster_index")),
            "precursor_mz": round(float(recorded["precursor_mz"]), 4),
            "GNPS_family_id": str(recorded.get("gnps_family")),
            "five_strongest_real_spectral_peaks": top_peaks,
            "nearest_neighbour_by_method": matches,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tahlan_streptomyces")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    import common as C
    from annotations import validated_annotations
    from exp_e_fixed_fusion import fusion_scores
    from exp_f_external_validation import check_cached_models

    p = C.run_paths(args.run, "protocol")
    check_cached_models(p, C)
    meta, embeddings = C.load_metadata_and_embeddings(p)
    ms2 = C.load_ms2deepscore(p)
    if ms2 is None or len(ms2) != len(embeddings):
        raise SystemExit("Missing or misaligned MS2DeepScore cached embeddings")
    selected, _ = validated_annotations(meta)
    if len(selected) < 20:
        raise SystemExit("Not enough chemical references for educational example")
    idx = selected["row"].to_numpy(dtype=int)
    score_matrices = fusion_scores(embeddings[idx], ms2[idx], 0.5)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [gen.GetFingerprint(mol) for mol in selected["mol"]]
    tanimoto = np.array([DataStructs.BulkTanimotoSimilarity(fp, fingerprints) for fp in fingerprints])
    all_spectra = C.read_clean_spectra(p, meta)
    examples = make_examples(selected, meta, all_spectra, score_matrices, tanimoto, args.count)

    payload = {
        "run": args.run,
        "data_origin": "Stored GNPS spectra, validated putative GNPS library annotations, cached published-model embeddings",
        "sampling_rule": "First distinct structural connectivity keys in alphabetical order, not best-performing cases",
        "not_training": True,
        "not_independently_identified_structures": True,
        "n_independent_putative_structures_in_run": len(selected),
        "examples": examples,
        "interpretation": [
            "m/z and intensities are real source peaks; only illustrative subset shown.",
            "Compound names and structural similarities derive from PUTATIVE GNPS library annotations.",
            "The nearest neighbour can differ by model; a high spectral similarity does not prove chemical identity.",
            "A nearest-neighbour example does not prove one model is best; use the aggregate benchmarks for that.",
        ],
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
