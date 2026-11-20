"""Cross-dataset evaluation of the *pre-specified* fixed 50/50 fusion.

Tries to evaluate a pre-defined external dataset (Chevrette) using the exact
same frozen DreaMS and MS2DeepScore checkpoints/peak preprocessing as the
Tahlan evaluation. Chemical connectivity (first InChIKey block) appearing
in Tahlan is removed from the external test to avoid shared structures.

This is an EXTERNAL-DATASET descriptive test, NOT a claim that we have
independent experimental structure confirmation or no pretraining leakage.
The external dataset has few putatively annotated compounds: results may
be inconclusive. Never tune the fusion weight on this evaluation set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exp_e_fixed_fusion import (
    evaluate, fusion_scores, paired_jackknife_delta,
    paired_top1_bootstrap_delta,
)


def disjoint_external_keys(reference_keys, external_keys):
    """Retain external rows NOT seen as molecular connectivity in reference."""
    reference = set(reference_keys)
    keys = list(external_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("External compounds contain duplicate InChIKey connectivity")
    kept = np.asarray([i for i, key in enumerate(keys) if key not in reference], dtype=int)
    overlap = sorted(set(keys) & reference)
    return kept, overlap


def check_cached_models(p, C):
    if not C.embeddings_up_to_date(p.embeddings_info, p.clean_mgf, expected={
        "model": "DreaMS_embedding (pre-trained, no fine-tuning)",
        "dreams_commit": C.DREAMS_COMMIT,
    }):
        raise ValueError(f"{p.name}: DreaMS cache does not match input spectra or model revision")
    if not C.embeddings_up_to_date(p.ms2ds_info, p.clean_mgf, expected={
        "input_peaks": C.INPUT_PEAKS,
        "model": "MS2DeepScore pre-trained (Zenodo 17826815)",
    }):
        raise ValueError(f"{p.name}: MS2DeepScore cache does not match input spectra or model revision")


def selected_run(name, C, validated_annotations):
    p = C.run_paths(name, "protocol")
    check_cached_models(p, C)
    metadata, dreams = C.load_metadata_and_embeddings(p)
    ms2ds = C.load_ms2deepscore(p)
    if ms2ds is None or len(ms2ds) != len(dreams):
        raise ValueError("Embedding arrays have mismatched spectra")
    selected, _ = validated_annotations(metadata)
    if selected["key"].nunique() != len(selected):
        raise ValueError("Chemical connectivity must be unique after filtering")
    return p, selected, dreams, ms2ds


def cross_dataset_report(reference_keys, external_keys, dreams, ms2ds,
                         tanimoto, *, min_external=20):
    """Evaluate only external compounds not present in the Tahlan set.

    IMPORTANT: structure matrix is expected to align to external keys and
    applies before filtering; model weights cannot be tuned here.
    """
    indices, overlap = disjoint_external_keys(reference_keys, external_keys)
    n = len(indices)
    r = {
        "external_compounds_before_overlap_filter": len(external_keys),
        "overlap_with_reference_count": len(overlap),
        "overlap_connectivity_keys": overlap,
        "external_compounds_after_overlap_filter": n,
        "minimum_for_exploratory_evaluation": min_external,
        "sufficient_for_exploratory_evaluation": n >= min_external,
        "status": "inconclusive: small external chemical reference",
        "scores": None,
        "paired_exploratory_uncertainty": None,
    }
    if n < min_external:
        return r
    if len(dreams) != len(external_keys) or len(ms2ds) != len(external_keys):
        raise ValueError("Model data and external spectrum labels must align")
    t = np.asarray(tanimoto, dtype=float)[np.ix_(indices, indices)]
    similarities = fusion_scores(dreams[indices], ms2ds[indices], 0.5)
    scores = evaluate(similarities, t, [external_keys[i] for i in indices])
    uncertainty = {}
    for baseline in ("DreaMS", "MS2DeepScore"):
        uncertainty[f"fusion_minus_{baseline}"] = {
            "spearman": paired_jackknife_delta(similarities, t, "fixed_50_50_fusion", baseline),
            "top1": paired_top1_bootstrap_delta(similarities, t, "fixed_50_50_fusion", baseline, rounds=2000),
        }
    r.update({
        "status": "evaluated, but external annotations are putative and sample size is small",
        "scores": scores,
        "paired_exploratory_uncertainty": uncertainty,
    })
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="tahlan_streptomyces")
    parser.add_argument("--external", default="chevrette_insect")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reference == args.external:
        raise SystemExit("Reference and external runs must be different")

    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from annotations import validated_annotations
    import common as C

    p_ref, reference, _, _ = selected_run(args.reference, C, validated_annotations)
    p_ext, external, dreams, ms2ds = selected_run(args.external, C, validated_annotations)
    if len(reference) < 100:
        raise SystemExit("Reference run has <100 chemically checked distinct compounds")
    n = len(external)
    idx = external["row"].to_numpy(dtype=int)
    if len(set(idx)) != len(idx):
        raise SystemExit("Cannot align repeated external metadata rows")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(mol) for mol in external["mol"]]
    tanimoto = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], dtype=float)

    report = cross_dataset_report(
        reference["key"].tolist(), external["key"].tolist(),
        dreams[idx], ms2ds[idx], tanimoto,
    )
    report.update({
        "reference_run": args.reference,
        "external_run": args.external,
        "reference_massive": C.read_json(p_ref.run_info).get("MassIVE dataset (PoDP)"),
        "external_massive": C.read_json(p_ext.run_info).get("MassIVE dataset (PoDP)"),
        "fusion_weight": "pre-specified 0.5 / 0.5; not tuned on either dataset",
        "checkpoint_identity": {
            "dreams_commit": C.DREAMS_COMMIT,
            "ms2deepscore": "Zenodo 17826815",
        },
        "limits": [
            "Different MassIVE datasets do NOT by themselves prove biological independence.",
            "First-block InChIKey overlap is excluded across the two evaluation sets.",
            "Unobserved overlap with public model PRETRAINING data is untested.",
            "GNPS library hits are putative, even after strict chemical matching.",
            "External n is small; jackknife/bootstrap intervals are exploratory.",
            "No model was trained or modified. No BGC–MS/MS predictive task was evaluated.",
        ],
    })
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
