"""Compare our DreaMS model trained from scratch with the published DreaMS.

The fair comparison is between two self-supervised models read the same way
(scripts/embed_ssl.py): the published DreaMS before fine-tuning (dreams_ssl)
and ours (for example scratch_small). The published fine-tuned DreaMS
embedding, MS2DeepScore and the cosine baselines are shown for context.

Same tests as Experiments A and B, on the protocol spectra of every run:
  A1  family recovery@5 against the GNPS molecular families
  B1  Spearman between spectral similarity and structural similarity (Tanimoto)
  B2  analog search: Tanimoto of the most similar spectrum of another molecule
B counts only for runs with at least 100 different molecules.

Usage:
  python scripts/compare_models.py --models scratch_small
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.similarity import ModifiedCosine
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

import common as C
from exp_a_families import family_recovery, knn
from annotations import validated_annotations
from exp_b_structures import MIN_UNIQUE_STRUCTURES, fragment_tolerance, spearman_difference_ci, top1_values

warnings.filterwarnings("ignore")
OUT_DIR = C.EXPERIMENT_DIR / "model_comparison"
REFERENCE = "dreams_ssl"   # the published DreaMS, self-supervised only


def load_named(p, name):
    """Unit-length embeddings saved by embed_ssl.py, or None if missing or out of date."""
    path, info = p.data / f"{name}_embeddings_protocol.npy", p.data / f"{name}_embeddings_protocol.json"
    if not (path.exists() and info.exists()):
        return None
    if not C.embeddings_up_to_date(info, p.clean_mgf):
        print(f"{p.name}: {name} embeddings are out of date; skipped.")
        return None
    emb = np.load(path).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def representations(p, models):
    meta, emb = C.load_metadata_and_embeddings(p)
    reps = {}
    for name in [REFERENCE, *models]:
        e = load_named(p, name)
        if e is not None:
            reps[name] = e
    reps["DreaMS embedding (published, fine-tuned)"] = emb
    ms2ds = C.load_ms2deepscore(p)
    if ms2ds is not None:
        reps["MS2DeepScore"] = ms2ds
    return meta, reps


def compare_run(run, models, k=5):
    p = C.run_paths(run, "protocol")
    if not p.metadata.exists():
        return []
    meta, reps = representations(p, models)
    rows = []
    spectra = [C.top_peaks(s) for s in C.read_clean_spectra(p, meta)]  # same peaks as DreaMS

    # A1. Family recovery@5
    labels, sizes = C.family_labels(meta), meta["family_size"].to_numpy()
    queries = np.where(sizes >= 3)[0]
    n = len(meta)
    binned = C.binned_vectors(spectra)
    scores_a = {name: e for name, e in reps.items()}
    scores_a["binned spectrum cosine"] = binned
    recovery = {}
    for name, e in scores_a.items():
        nn, _ = knn(lambda a, b, e=e: e[a:b] @ e.T, n, k)
        recovery[name] = family_recovery(nn, labels, sizes, queries, k)
        rows.append((run, "A1 family recovery@5", name, round(float(recovery[name].mean()), 3)))
    for name in models:
        if name in recovery and REFERENCE in recovery:
            rows.append((run, "A1 family recovery@5", f"{name} minus {REFERENCE} (95% CI)",
                         C.ci_text(*C.bootstrap_mean_ci(recovery[name] - recovery[REFERENCE], labels[queries]))))

    # B. Structure, only library-annotated spectra
    selected, _ = validated_annotations(meta)
    idx, keys, mols = selected["row"].tolist(), selected["key"].to_numpy(), selected["mol"].tolist()
    n_structures = len(set(keys))
    rows.append((run, "B unique structures", "", n_structures))
    if n_structures < MIN_UNIQUE_STRUCTURES:
        return rows
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(m) for m in mols]
    tanimoto = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    scores_b = {name: e[idx] @ e[idx].T for name, e in reps.items()}
    specs = [Spectrum(mz=spectra[i]["mz"].astype(float),
                      intensities=(spectra[i]["intensity"] / spectra[i]["intensity"].max()).astype(float),
                      metadata={"precursor_mz": float(spectra[i]["precursor_mz"])}) for i in idx]
    scores_b["modified cosine"] = ModifiedCosine(tolerance=fragment_tolerance(p)).matrix(
        specs, specs, is_symmetric=True)["score"]
    iu = np.triu_indices(len(idx), k=1)
    blocked = np.eye(len(idx), dtype=bool) | (keys[:, None] == keys[None, :])
    top1 = {}
    for name, s in scores_b.items():
        rows.append((run, "B1 Spearman with Tanimoto", name, round(float(spearmanr(tanimoto[iu], s[iu]).correlation), 3)))
        values, ok = top1_values(s, tanimoto, blocked)
        top1[name] = (values, ok)
        rows.append((run, "B2 analog search top-1 Tanimoto", name, round(float(values[ok].mean()), 3)))
    for name in models:
        if name in scores_b and REFERENCE in scores_b:
            rows.append((run, "B1 Spearman with Tanimoto", f"{name} minus {REFERENCE} (95% CI)",
                         C.ci_text(*spearman_difference_ci(tanimoto, scores_b[name], scores_b[REFERENCE], keys))))
            ok = top1[REFERENCE][1]
            rows.append((run, "B2 analog search top-1 Tanimoto", f"{name} minus {REFERENCE} (95% CI)",
                         C.ci_text(*C.bootstrap_mean_ci((top1[name][0] - top1[REFERENCE][0])[ok], keys[ok]))))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="names given to embed_ssl.py, e.g. scratch_small")
    parser.add_argument("--runs", nargs="*", default=list(C.RUNS))
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        print(f"Comparing models on {run}...")
        rows += compare_run(run, args.models)
    df = pd.DataFrame(rows, columns=["run", "metric", "representation", "value"])
    df["value"] = df["value"].astype(str)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(OUT_DIR / "model_comparison.parquet", index=False)
    table = df.pivot_table(index=["metric", "representation"], columns="run", values="value", aggfunc="first").reset_index()
    text = "\n".join([
        "# Our DreaMS trained from scratch vs the published DreaMS",
        "",
        f"Generated by `scripts/compare_models.py`. Protocol spectra. `{REFERENCE}` is the published DreaMS "
        "before fine-tuning, read the same way as our models (precursor token, last layer). "
        "Differences have a 95% bootstrap interval; an interval that contains 0 means no detectable difference. "
        f"B is computed only for runs with at least {MIN_UNIQUE_STRUCTURES} different molecules.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
        *["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in table.to_numpy()],
    ])
    (OUT_DIR / "RESULTS_model_comparison.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
