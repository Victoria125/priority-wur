"""Step 4 (Experiment B). Does DreaMS similarity follow chemical structure?

Only spectra with a GNPS library hit that has a structure (SMILES) are used.
For every pair of these spectra we compare:
  - structural similarity of the two molecules (Tanimoto, Morgan fingerprints)
  - DreaMS similarity (cosine of the embeddings)
  - modified cosine (the spectral score GNPS uses to build its network)
  - binned spectrum cosine (a simple spectral baseline)
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr

import common as C

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def molecule(smiles):
    """SMILES -> RDKit molecule without salts / counter ions (None if invalid)."""
    if not isinstance(smiles, str) or not smiles.strip() or smiles.strip().upper() in ("N/A", "NA"):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return rdMolStandardize.LargestFragmentChooser().choose(mol)


def fragment_tolerance(p, default=0.02):
    """Use the same fragment tolerance as the GNPS job, if we know it."""
    try:
        return float(C.read_json(p.run_info)["fragment tolerance used by GNPS (Da)"])
    except Exception:
        return default


def top1_tanimoto(similarity, tanimoto, blocked):
    """For each spectrum, take its most similar other spectrum (by a spectral
    score) and return the Tanimoto of that pair. Blocked pairs are skipped."""
    s = similarity.astype(float).copy()
    s[blocked] = -np.inf
    has_candidate = np.isfinite(s.max(axis=1))
    best = s.argmax(axis=1)
    values = tanimoto[np.arange(len(s)), best][has_candidate]
    return float(values.mean()), int(has_candidate.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--min-library-score", type=float, default=0.0,
                        help="only use library hits with at least this GNPS score (MQScore)")
    args = parser.parse_args()

    p = C.run_paths(args.run)
    meta, emb = C.load_metadata_and_embeddings(p)
    spectra = C.read_mgf(p.clean_mgf)  # same order as meta
    assert len(spectra) == len(meta)

    mols = [molecule(s) for s in meta["smiles"]]
    score_ok = meta["library_score"].fillna(-1).to_numpy() >= args.min_library_score
    idx = [i for i, m in enumerate(mols) if m is not None and score_ok[i]]
    rows = [("annotated spectra used", len(idx), "GNPS library hit with a valid structure")]
    if len(idx) < 20:
        C.save_summary(rows + [("note", "too few annotated spectra for this test", "")],
                       p.results / "exp_b_summary.parquet")
        print("Too few annotated spectra for Experiment B. Skipping.")
        return

    # Structural similarity
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mols[i]) for i in idx]
    keys = np.array([Chem.MolToInchiKey(mols[i])[:14] for i in idx])  # same 2D structure -> same key
    tanimoto = np.array([DataStructs.BulkTanimotoSimilarity(fp, fingerprints) for fp in fingerprints])
    n = len(idx)
    rows.append(("unique structures", int(len(set(keys))), "first block of the InChIKey"))

    # DreaMS similarity
    e = emb[idx]
    dreams = e @ e.T

    # Binned spectrum cosine
    binned_vectors = C.binned_vectors([spectra[i] for i in idx])
    binned = binned_vectors @ binned_vectors.T

    # Modified cosine (matchms), with the fragment tolerance of the GNPS job
    tolerance = fragment_tolerance(p)
    specs = []
    for i in idx:
        s = spectra[i]
        specs.append(Spectrum(
            mz=s["mz"].astype(float),
            intensities=(s["intensity"] / s["intensity"].max()).astype(float),
            metadata={"precursor_mz": float(s["precursor_mz"])},
        ))
    print(f"Computing modified cosine for {n * (n - 1) // 2:,} pairs (tolerance {tolerance} Da)...")
    modcos = ModifiedCosine(tolerance=tolerance).matrix(specs, specs, is_symmetric=True)["score"]
    rows.append(("fragment tolerance for modified cosine (Da)", tolerance, "taken from the GNPS job"))

    scores = {"DreaMS": dreams, "modified cosine": modcos, "binned spectrum cosine": binned}

    # 1. Correlation with structure over all pairs
    iu = np.triu_indices(n, k=1)
    t = tanimoto[iu]
    rho = {name: spearmanr(t, s[iu]).correlation for name, s in scores.items()}
    for name in scores:
        rows.append((f"B1 Spearman (Tanimoto vs {name})", round(float(rho[name]), 3), f"{len(t):,} pairs"))

    # 2. Top-1 search: is the most similar spectrum also a similar molecule?
    self_mask = np.eye(n, dtype=bool)
    same_structure = keys[:, None] == keys[None, :]
    for label, blocked in [("any molecule", self_mask), ("other molecules only (analog search)", self_mask | same_structure)]:
        t_blocked = tanimoto.astype(float).copy()
        t_blocked[blocked] = np.nan
        random_level = float(np.nanmean(t_blocked))
        best_level = float(np.nanmax(t_blocked, axis=1)[~np.all(np.isnan(t_blocked), axis=1)].mean())
        for name, s in scores.items():
            mean_t, n_queries = top1_tanimoto(s, tanimoto, blocked)
            rows.append((f"B2 top-1 neighbour Tanimoto, {label}: {name}", round(mean_t, 3), f"{n_queries} query spectra"))
        rows += [
            (f"B2 top-1 neighbour Tanimoto, {label}: random", round(random_level, 3), "chance level"),
            (f"B2 top-1 neighbour Tanimoto, {label}: best possible", round(best_level, 3), "upper limit"),
        ]

    # Picture
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (name, s) in zip(axes, scores.items()):
        ax.hexbin(s[iu], t, gridsize=40, bins="log", cmap="viridis", mincnt=1)
        ax.set_xlabel(name)
        ax.set_title(f"{name} (Spearman {rho[name]:.2f})")
    axes[0].set_ylabel("structural similarity (Tanimoto)")
    fig.suptitle(f"B. Spectral similarity vs structural similarity (library-annotated spectra), {args.run}")
    fig.tight_layout()
    fig.savefig(p.results / "exp_b_structure.png", dpi=150)
    plt.close(fig)

    summary = C.save_summary(rows, p.results / "exp_b_summary.parquet")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
