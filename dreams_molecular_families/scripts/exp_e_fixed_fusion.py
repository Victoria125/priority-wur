"""Exploratory, pre-specified 50/50 fusion of existing spectral embeddings.

No model is trained or tuned. DreaMS/MS2DeepScore input spectra and stored
embeddings must be the ones already validated by common.py and annotations.py.
Main endpoint: Spearman with chemical-structure Morgan Tanimoto for unique
library-annotated compounds. The annotations remain putative.

DO NOT treat a higher point estimate as proven improvement, and do not
present this as BGC–MS/MS linking. For formal comparisons, require independent
data and appropriately controlled evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("Need >= 3 spectra and 2-D embedding arrays")
    if not np.isfinite(x).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length embedding cannot be normalized")
    x = x / norms
    return x @ x.T


def fusion_scores(dreams: np.ndarray, ms2ds: np.ndarray, alpha: float = 0.5):
    """Fixed score-level fusion. No joint training or shared embedding space."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if len(dreams) != len(ms2ds):
        raise ValueError("Models must represent the SAME spectra in the SAME order")
    d = cosine_matrix(dreams)
    m = cosine_matrix(ms2ds)
    return {"DreaMS": d, "MS2DeepScore": m,
            "fixed_50_50_fusion": alpha * d + (1 - alpha) * m}


def evaluate(scores: dict[str, np.ndarray], tanimoto: np.ndarray, keys):
    """Descriptive evaluation, one row per unique molecular connectivity."""
    t = np.asarray(tanimoto, dtype=float)
    n = len(keys)
    if n < 20 or t.shape != (n, n):
        raise ValueError("Need >= 20 independently selected compounds and square structural similarities")
    if len(set(keys)) != n:
        raise ValueError("Duplicate molecule connectivity: select one spectrum per molecule before evaluation")
    if np.any(~np.isfinite(t)) or np.any((t < 0) | (t > 1)):
        raise ValueError("Invalid Tanimoto values")
    idx = np.triu_indices(n, 1)
    if np.std(t[idx]) == 0:
        raise ValueError("Chemical reference is constant; Spearman undefined")
    out = {}
    for method, mat in scores.items():
        s = np.asarray(mat, dtype=float)
        if s.shape != (n, n) or np.any(~np.isfinite(s)):
            raise ValueError("Score matrix has missing/nonfinite values")
        if not np.allclose(s, s.T, atol=1e-6):
            raise ValueError("Score matrix is not symmetric")
        rho = float(spearmanr(s[idx], t[idx]).correlation)
        if not np.isfinite(rho):
            raise ValueError("Undefined correlation for " + method)
        blocked = s.copy()
        np.fill_diagonal(blocked, -np.inf)
        best = blocked.argmax(axis=1)
        top1 = float(t[np.arange(n), best].mean())
        out[method] = {"spearman_with_tanimoto": rho, "top1_analog_tanimoto": top1}
    return out



def paired_jackknife_delta(scores: dict[str, np.ndarray],
                           target: np.ndarray, first: str, other: str) -> dict:
    """Approximate molecule-delete-one jackknife interval for delta Spearman.

    This accounts for shared molecule membership of pairwise comparisons better
    than bootstrapping the correlated spectrum pairs. It is not an independent
    laboratory validation or a rigorous CI under molecular-family dependence.
    """
    a = np.asarray(scores[first], dtype=float)
    b = np.asarray(scores[other], dtype=float)
    t = np.asarray(target, dtype=float)
    n = len(a)
    if n < 20 or a.shape != (n, n) or b.shape != (n, n) or t.shape != (n, n):
        raise ValueError("Jackknife requires >= 20 matched square matrices")
    def difference(indices):
        ii = np.ix_(indices, indices)
        tri = np.triu_indices(len(indices), 1)
        tar = t[ii][tri]
        return float(spearmanr(a[ii][tri], tar).correlation -
                     spearmanr(b[ii][tri], tar).correlation)
    full = difference(np.arange(n))
    deletion = np.array([
        difference(np.r_[0:i, i+1:n])
        for i in range(n)
    ])
    if not np.all(np.isfinite(deletion)):
        raise ValueError("Undefined leave-one-molecule-out Spearman")
    se = float(np.sqrt((n - 1) / n * np.square(deletion - deletion.mean()).sum()))
    return {
        "delta": full,
        "approx_95pct_interval": [full - 1.96*se, full + 1.96*se],
        "method": "delete-one-molecule jackknife; normal approximation; exploratory",
        "interval_includes_zero": bool(full - 1.96*se <= 0 <= full + 1.96*se),
    }


def paired_top1_bootstrap_delta(scores: dict[str, np.ndarray],
                                target: np.ndarray, first: str, other: str,
                                seed: int = 3402, rounds: int = 2000) -> dict:
    """Query-level bootstrap on fixed candidate pools, NOT external validation."""
    n = len(target)
    if n < 20 or rounds < 50:
        raise ValueError("Need >=20 queries and >=50 bootstrap rounds")
    def per_query(name):
        sim = np.asarray(scores[name], dtype=float).copy()
        np.fill_diagonal(sim, -np.inf)
        return target[np.arange(n), sim.argmax(axis=1)]
    differences = per_query(first) - per_query(other)
    rng = np.random.default_rng(seed)
    estimates = np.array([
        differences[rng.integers(0, n, size=n)].mean()
        for _ in range(rounds)
    ])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "delta": float(differences.mean()),
        "approx_95pct_interval": [float(low), float(high)],
        "method": "query-molecule bootstrap; fixed candidate pool; exploratory",
        "interval_includes_zero": bool(low <= 0 <= high),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tahlan_streptomyces")
    parser.add_argument("--variant", default="protocol", choices=("protocol", "all_kept"))
    parser.add_argument("--output", type=Path, help="Optional JSON output")
    args = parser.parse_args()

    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    import common as C
    from annotations import validated_annotations

    p = C.run_paths(args.run, args.variant)
    meta, dreams = C.load_metadata_and_embeddings(p)
    ms2ds = C.load_ms2deepscore(p)
    if ms2ds is None:
        raise SystemExit("Missing matching MS2DeepScore embeddings; run embed_ms2deepscore.py first")
    if len(ms2ds) != len(meta):
        raise SystemExit("MS2DeepScore shape does not match metadata")

    selected, _ = validated_annotations(meta)
    if len(selected) < 100:
        raise SystemExit("Too few compounds for primary structural benchmarking (<100)")
    if selected["key"].nunique() != len(selected):
        raise SystemExit("Structure selection must yield one spectrum per independent compound")

    idx = selected["row"].to_numpy(dtype=int)
    scores = fusion_scores(dreams[idx], ms2ds[idx])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(mol) for mol in selected["mol"]]
    tanimoto = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], dtype=float)
    measurements = evaluate(scores, tanimoto, selected["key"].tolist())
    paired_comparisons = {}
    for baseline in ("DreaMS", "MS2DeepScore"):
        paired_comparisons[f"fusion_minus_{baseline}"] = {
            "spearman": paired_jackknife_delta(scores, tanimoto, "fixed_50_50_fusion", baseline),
            "top1": paired_top1_bootstrap_delta(scores, tanimoto, "fixed_50_50_fusion", baseline),
        }
    result = {
        "dataset": args.run, "variant": args.variant,
        "independent_compounds": len(selected),
        "pairs": len(selected) * (len(selected)-1) // 2,
        "fusion_weight": {"DreaMS": 0.5, "MS2DeepScore": 0.5},
        "training": "none; fixed weights pre-specified",
        "scores": measurements,
        "paired_exploratory_uncertainty": paired_comparisons,
        "warnings": [
            "Library annotations remain putative even after chemical QC.",
            "No external hold-out or independent lab validation in this experiment.",
            "Jackknife and query bootstrap intervals are exploratory under dependent chemical families.",
            "No claim of independent-lab superiority from point estimates or exploratory intervals.",
            "Neither model directly predicts BGC–metabolite links.",
            "Test compounds may overlap pretrained public library datasets.",
        ],
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
