"""Step 3 (Experiment A). Do DreaMS embeddings group spectra like the GNPS
molecular families?

Careful: GNPS families are built from modified cosine similarity, so this
experiment measures agreement with GNPS, not "truth". Experiment B uses
chemical structures as an independent reference.
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

import common as C

warnings.filterwarnings("ignore")

# Similarity thresholds tried for the DreaMS network (A4).
SWEEP_THRESHOLDS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def knn(score_block, n, k, chunk=2048):
    """k nearest neighbours for every spectrum (the spectrum itself excluded).

    score_block(start, stop) must return a (stop - start, n) matrix where a
    higher value means 'more similar'. Working in blocks keeps memory low.
    """
    k = min(k, n - 1)
    idx_out = np.empty((n, k), dtype=np.int64)
    score_out = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        scores = np.array(score_block(start, stop), dtype=np.float64)
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx_out[start:stop] = np.take_along_axis(part, order, axis=1)
        score_out[start:stop] = np.take_along_axis(part_scores, order, axis=1)
    return idx_out, score_out


def family_recovery(neighbours, labels, sizes, queries, k):
    """For each query spectrum: (same-family neighbours) / (best possible number).
    1.0 means all possible family partners are among the k nearest neighbours."""
    hits = (labels[neighbours[queries, :k]] == labels[queries, None]).sum(axis=1)
    best = np.minimum(k, sizes[queries] - 1)
    return hits / best


def knn_network(neighbours, similarity, threshold):
    """Link each spectrum to its nearest neighbours with similarity >= threshold
    (like the DreaMS networking tutorial). Returns the unique edges (i < j), a
    family label per spectrum (singletons get a unique negative label) and the
    size of the family of each spectrum."""
    n = len(neighbours)
    src = np.repeat(np.arange(n), neighbours.shape[1])
    keep = similarity.ravel() >= threshold
    pairs = np.sort(np.stack([src[keep], neighbours.ravel()[keep]], axis=1), axis=1)
    pairs = np.unique(pairs, axis=0) if len(pairs) else pairs.reshape(0, 2)
    adjacency = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, component = connected_components(adjacency, directed=False)
    component_size = np.bincount(component)[component]
    labels = np.where(component_size >= 2, component, -(np.arange(n) + 1))
    return pairs, labels, component_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--k", type=int, default=5, help="neighbours for the family test")
    parser.add_argument("--net-k", type=int, default=3, help="neighbours per spectrum in the DreaMS network")
    parser.add_argument("--net-threshold", type=float, default=0.7, help="minimum DreaMS similarity for a network edge")
    parser.add_argument("--min-family-size", type=int, default=3)
    args = parser.parse_args()

    p = C.run_paths(args.run)
    meta, emb = C.load_metadata_and_embeddings(p)
    n = len(meta)
    labels = C.family_labels(meta)
    sizes = meta["family_size"].to_numpy()
    pmz = meta["precursor_mz"].to_numpy()
    rows = []

    # ------------------------------------------------------------------
    # A1. Are spectra from the same GNPS family close in DreaMS space?
    # ------------------------------------------------------------------
    queries = np.where(sizes >= args.min_family_size)[0]
    if len(queries) == 0:
        raise SystemExit(f"No GNPS family with >= {args.min_family_size} members after cleaning.")
    nn_dreams, _ = knn(lambda a, b: emb[a:b] @ emb.T, n, args.k)
    # Baseline 1: neighbours chosen only by precursor m/z (closest masses).
    nn_mass, _ = knn(lambda a, b: -np.abs(pmz[a:b, None] - pmz[None, :]), n, args.k)
    # Baseline 2: cosine of binned spectra (a spectral score without learning).
    binned = C.binned_vectors(C.read_mgf(p.clean_mgf))
    nn_binned, _ = knn(lambda a, b: binned[a:b] @ binned.T, n, args.k)

    rec_dreams = family_recovery(nn_dreams, labels, sizes, queries, args.k)
    rec_binned = family_recovery(nn_binned, labels, sizes, queries, args.k)
    rec_mass = family_recovery(nn_mass, labels, sizes, queries, args.k)
    expected_hits = args.k * (sizes[queries] - 1) / (n - 1)  # if neighbours were random
    rec_random = expected_hits / np.minimum(args.k, sizes[queries] - 1)

    rows += [
        ("spectra used", n, "after the DreaMS quality rules"),
        (f"spectra in GNPS families with >= {args.min_family_size} members", len(queries), "query spectra for A1"),
        (f"A1 family recovery@{args.k}: DreaMS", round(rec_dreams.mean(), 3), "1.0 = all family partners in the top k"),
        (f"A1 family recovery@{args.k}: binned spectrum cosine", round(rec_binned.mean(), 3), f"baseline, {C.BIN_WIDTH:g} Da bins"),
        (f"A1 family recovery@{args.k}: precursor m/z only", round(rec_mass.mean(), 3), "baseline"),
        (f"A1 family recovery@{args.k}: random", round(rec_random.mean(), 3), "chance level"),
    ]

    # ------------------------------------------------------------------
    # A2. DreaMS similarity of GNPS network edges vs random pairs.
    # ------------------------------------------------------------------
    edges = pd.read_parquet(p.edges)
    row_of = pd.Series(meta["row"].to_numpy(), index=meta["cluster_index"].to_numpy())
    r1 = row_of.loc[edges["id1"]].to_numpy()
    r2 = row_of.loc[edges["id2"]].to_numpy()
    edge_sim = np.einsum("ij,ij->i", emb[r1], emb[r2])

    rng = np.random.default_rng(0)
    n_random = max(5000, len(edges))
    a = rng.integers(0, n, n_random)
    b = rng.integers(0, n, n_random)
    ok = a != b
    random_sim = np.einsum("ij,ij->i", emb[a[ok]], emb[b[ok]])

    if len(edges) >= 3:
        rho = spearmanr(edges["gnps_cosine"], edge_sim).correlation
        rows += [
            ("A2 median DreaMS similarity: GNPS edges", round(float(np.median(edge_sim)), 3), f"{len(edges)} edges"),
            ("A2 median DreaMS similarity: random pairs", round(float(np.median(random_sim)), 3), f"{ok.sum()} pairs"),
            ("A2 Spearman (GNPS cosine vs DreaMS similarity) on GNPS edges", round(float(rho), 3), ""),
        ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(edge_sim.min(initial=1), random_sim.min()), 1, 50)
    ax.hist(random_sim, bins=bins, density=True, alpha=0.6, label="random pairs")
    if len(edge_sim):
        ax.hist(edge_sim, bins=bins, density=True, alpha=0.6, label="pairs connected in GNPS network")
    ax.set_xlabel("DreaMS similarity (cosine)")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title(f"A2. DreaMS similarity: GNPS edges vs random pairs\n{args.run}", fontsize=10)
    fig.tight_layout()
    fig.savefig(p.results / "exp_a_similarity_hist.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # A3. Build a DreaMS molecular network (k-NN graph, like the DreaMS
    # tutorial) and compare its families with the GNPS families.
    # ------------------------------------------------------------------
    nn_net, nn_net_sim = knn(lambda a_, b_: emb[a_:b_] @ emb.T, n, args.net_k)
    pairs, dreams_labels, component_size = knn_network(nn_net, nn_net_sim, args.net_threshold)
    in_gnps_family = sizes >= 2

    rows += [
        ("A3 DreaMS network settings", f"k={args.net_k}, similarity >= {args.net_threshold}", "as in the DreaMS networking tutorial"),
        ("A3 DreaMS network edges", len(pairs), ""),
        ("A3 DreaMS families (size >= 2)", int(len(np.unique(dreams_labels[component_size >= 2]))), ""),
        ("A3 largest DreaMS family", int(component_size.max()), "spectra"),
        ("A3 agreement with GNPS families (AMI)", round(adjusted_mutual_info_score(labels, dreams_labels), 3), "1 = identical, 0 = chance"),
        ("A3 agreement with GNPS families (ARI)", round(adjusted_rand_score(labels, dreams_labels), 3), "1 = identical, 0 = chance"),
    ]
    if in_gnps_family.sum() >= 2:
        rows.append((
            "A3 agreement (AMI), only spectra in GNPS families",
            round(adjusted_mutual_info_score(labels[in_gnps_family], dreams_labels[in_gnps_family]), 3),
            f"{int(in_gnps_family.sum())} spectra",
        ))

    gnps_pairs = {tuple(sorted(pair)) for pair in zip(r1, r2)}
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        in_family_i, in_family_j = sizes[i] >= 2, sizes[j] >= 2
        category = np.where(
            ~(in_family_i & in_family_j), "involves a GNPS singleton",
            np.where(labels[i] == labels[j], "same GNPS family", "different GNPS families"),
        )
        for name in ["same GNPS family", "different GNPS families", "involves a GNPS singleton"]:
            rows.append((f"A3 DreaMS edges: {name}", int((category == name).sum()), f"{100 * (category == name).mean():.1f}%"))

        if gnps_pairs:
            dreams_pairs = {tuple(pair) for pair in pairs}
            recovered = len(gnps_pairs & dreams_pairs) / len(gnps_pairs)
            rows.append(("A3 GNPS edges also found by the DreaMS network", f"{100 * recovered:.1f}%", f"of {len(gnps_pairs)} GNPS edges"))

        # Links that GNPS did not make: hypotheses to look at by hand.
        new = pd.DataFrame({
            "cluster_1": meta["cluster_index"].to_numpy()[i],
            "cluster_2": meta["cluster_index"].to_numpy()[j],
            "dreams_similarity": np.round(np.einsum("ij,ij->i", emb[i], emb[j]), 3),
            "precursor_mz_1": np.round(pmz[i], 3),
            "precursor_mz_2": np.round(pmz[j], 3),
            "delta_mz": np.round(pmz[j] - pmz[i], 3),
            "library_hit_1": meta["compound_name"].to_numpy()[i],
            "library_hit_2": meta["compound_name"].to_numpy()[j],
            "category": category,
        })
        new = new[new["category"] == "involves a GNPS singleton"].sort_values("dreams_similarity", ascending=False)
        new.head(300).to_parquet(p.results / "exp_a_new_links.parquet", index=False)

    # ------------------------------------------------------------------
    # A4. The same DreaMS network with other similarity thresholds. With a
    # low threshold many spectra end up in one giant family, which hides
    # the agreement with GNPS.
    # ------------------------------------------------------------------
    sweep = []
    for threshold in SWEEP_THRESHOLDS:
        t_pairs, t_labels, t_size = knn_network(nn_net, nn_net_sim, threshold)
        t_dreams_pairs = {tuple(pair) for pair in t_pairs}
        sweep.append({
            "threshold": threshold,
            "edges": len(t_pairs),
            "families (size >= 2)": int(len(np.unique(t_labels[t_size >= 2]))),
            "spectra in families": int((t_size >= 2).sum()),
            "largest family": int(t_size.max()),
            "AMI, all spectra": round(adjusted_mutual_info_score(labels, t_labels), 3),
            "AMI, spectra in GNPS families": round(
                adjusted_mutual_info_score(labels[in_gnps_family], t_labels[in_gnps_family]), 3
            ) if in_gnps_family.sum() >= 2 else np.nan,
            "GNPS edges recovered (%)": round(
                100 * len(gnps_pairs & t_dreams_pairs) / len(gnps_pairs), 1
            ) if gnps_pairs else np.nan,
        })
    pd.DataFrame(sweep).to_parquet(p.results / "exp_a_network_sweep.parquet", index=False)

    # ------------------------------------------------------------------
    # UMAP picture: the 10 largest GNPS families in colour.
    # ------------------------------------------------------------------
    import umap

    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=3, c="lightgrey", label="other spectra")
    family_counts = meta.loc[sizes >= 2, "gnps_family"].value_counts().head(10)
    colours = plt.get_cmap("tab10")
    for c, (family, count) in enumerate(family_counts.items()):
        mask = (meta["gnps_family"].to_numpy() == family) & (sizes >= 2)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, color=colours(c), label=f"GNPS family {family} (n={count})")
    ax.set_title(f"DreaMS embeddings (UMAP), coloured by GNPS molecular family\n{args.run}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    fig.tight_layout()
    fig.savefig(p.results / "exp_a_umap.png", dpi=150)
    plt.close(fig)

    summary = C.save_summary(rows, p.results / "exp_a_summary.parquet")
    print(summary.to_string(index=False))
    print()
    print(pd.DataFrame(sweep).to_string(index=False))


if __name__ == "__main__":
    main()
