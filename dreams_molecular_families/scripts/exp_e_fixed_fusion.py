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
    result = {
        "dataset": args.run, "variant": args.variant,
        "independent_compounds": len(selected),
        "pairs": len(selected) * (len(selected)-1) // 2,
        "fusion_weight": {"DreaMS": 0.5, "MS2DeepScore": 0.5},
        "training": "none; fixed weights pre-specified",
        "scores": measurements,
        "warnings": [
            "Library annotations remain putative even after chemical QC.",
            "No external hold-out or independent lab validation in this experiment.",
            "No claim of statistical superiority from point estimates alone.",
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
