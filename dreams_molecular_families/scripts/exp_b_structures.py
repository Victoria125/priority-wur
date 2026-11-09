"""Step 4 (Experiment B). Does DreaMS similarity follow chemical structure?

Only spectra with a GNPS library hit that has a structure (SMILES) are used.
For every pair of these spectra we compare:
  - structural similarity of the two molecules: Tanimoto of Morgan
    fingerprints (main) and MHFP (Daniel Probst's MinHash fingerprint)
  - DreaMS similarity (cosine of the embeddings)
  - MS2DeepScore (cosine of its embeddings, if scripts/embed_ms2deepscore.py ran)
  - modified cosine (the spectral score GNPS uses to build its network)
  - binned spectrum cosine (a simple spectral baseline)

Differences between methods get a 95% interval from resampling molecules,
because spectra of the same molecule are not independent.
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from mhfp.encoder import MHFPEncoder
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr

import common as C

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# A dataset counts for conclusions only with at least this many different
# molecules. Rule set on 2026-11-09, after the first results: Duncan had 77
# in-domain structures and no method followed them (Spearman ~0.05).
MIN_UNIQUE_STRUCTURES = 100


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


def top1_values(similarity, tanimoto, blocked):
    """For each spectrum, take its most similar other spectrum (by a spectral
    score) and return the Tanimoto of that pair. Blocked pairs are skipped.
    Returns the values and which spectra had a candidate at all."""
    s = similarity.astype(float).copy()
    s[blocked] = -np.inf
    has_candidate = np.isfinite(s.max(axis=1))
    best = s.argmax(axis=1)
    return tanimoto[np.arange(len(s)), best], has_candidate


def top1_tanimoto(similarity, tanimoto, blocked):
    values, has_candidate = top1_values(similarity, tanimoto, blocked)
    return float(values[has_candidate].mean()), int(has_candidate.sum())


def mhfp_similarity(mols):
    """MHFP (MinHash fingerprint, 2048 hashes): share of equal hashes = estimated Jaccard."""
    encoder = MHFPEncoder(2048)
    fps = np.array([encoder.encode_mol(m) for m in mols])
    return np.array([(fps == fp).mean(axis=1) for fp in fps])


def spearman_difference_ci(structure, score_a, score_b, keys, n_boot=300, seed=0):
    """Spearman(structure, a) minus Spearman(structure, b), with a 95% interval
    from resampling molecules (all spectra of a molecule together). Pairs of a
    spectrum with its own copy are skipped."""
    rng = np.random.default_rng(seed)
    unique = np.unique(keys)
    rows_of = {k: np.where(keys == k)[0] for k in unique}

    def difference(idx):
        i, j = np.triu_indices(len(idx), k=1)
        keep = idx[i] != idx[j]
        a, b = idx[i][keep], idx[j][keep]
        t = structure[a, b]
        return spearmanr(t, score_a[a, b]).correlation - spearmanr(t, score_b[a, b]).correlation

    point = difference(np.arange(len(keys)))
    draws = [difference(np.concatenate([rows_of[k] for k in rng.choice(unique, len(unique))]))
             for _ in range(n_boot)]
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(low), float(high)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--variant", default="protocol", choices=list(C.VARIANTS))
    parser.add_argument("--min-library-score", type=float, default=0.0,
                        help="only use library hits with at least this GNPS score (MQScore)")
    args = parser.parse_args()

    p = C.run_paths(args.run, args.variant)
    p.results.mkdir(parents=True, exist_ok=True)
    meta, emb = C.load_metadata_and_embeddings(p)
    ms2ds = C.load_ms2deepscore(p)
    spectra = C.read_clean_spectra(p, meta)  # same order as meta

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
    mhfp = mhfp_similarity([mols[i] for i in idx])
    n = len(idx)
    n_structures = int(len(set(keys)))
    enough = n_structures >= MIN_UNIQUE_STRUCTURES
    rows += [
        ("unique structures", n_structures, "first block of the InChIKey"),
        ("B used for conclusions", "yes" if enough else "no",
         f"needs >= {MIN_UNIQUE_STRUCTURES} unique structures"),
    ]

    # Spectral similarities
    e = emb[idx]
    scores = {"DreaMS": e @ e.T}
    if ms2ds is not None:
        m = ms2ds[idx]
        scores["MS2DeepScore"] = m @ m.T
    binned_vectors = C.binned_vectors([spectra[i] for i in idx])
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
    scores["modified cosine"] = ModifiedCosine(tolerance=tolerance).matrix(specs, specs, is_symmetric=True)["score"]
    scores["binned spectrum cosine"] = binned_vectors @ binned_vectors.T
    rows.append(("fragment tolerance for modified cosine (Da)", tolerance, "taken from the GNPS job"))

    # 1. Correlation with structure over all pairs (Morgan Tanimoto, then MHFP)
    iu = np.triu_indices(n, k=1)
    t = tanimoto[iu]
    rho = {name: spearmanr(t, s[iu]).correlation for name, s in scores.items()}
    for name in scores:
        rows.append((f"B1 Spearman (Tanimoto vs {name})", round(float(rho[name]), 3), f"{len(t):,} pairs"))
    for name, s in scores.items():
        rows.append((f"B1 Spearman (MHFP vs {name})", round(float(spearmanr(mhfp[iu], s[iu]).correlation), 3), "MHFP Jaccard"))
    for other in [name for name in ("modified cosine", "MS2DeepScore") if name in scores]:
        rows.append((f"B1 Spearman with Tanimoto, DreaMS minus {other} (95% CI)",
                     C.ci_text(*spearman_difference_ci(tanimoto, scores["DreaMS"], scores[other], keys)),
                     "resampling molecules"))

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
        if label.startswith("other"):
            dreams_values, ok = top1_values(scores["DreaMS"], tanimoto, blocked)
            for other in [name for name in ("modified cosine", "MS2DeepScore") if name in scores]:
                other_values, _ = top1_values(scores[other], tanimoto, blocked)
                rows.append((f"B2 analog search, DreaMS minus {other} (95% CI)",
                             C.ci_text(*C.bootstrap_mean_ci((dreams_values - other_values)[ok], keys[ok])),
                             "resampling query molecules"))

    # Picture
    fig, axes = plt.subplots(1, len(scores), figsize=(4.7 * len(scores), 4.2), sharey=True)
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
