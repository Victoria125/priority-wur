"""Step 6 (Experiment C, first prototype). Can a small layer on top of the
frozen DreaMS embeddings follow chemical structure better, and can the labs
train it together without sharing their spectra?

Each run is one lab (one federated client). Only library-annotated spectra
are used, because the training target is the structural similarity
(Tanimoto) of the molecules of two spectra, as in MS2DeepScore.

Four ways to get a spectrum similarity are compared:
  - pre-trained DreaMS: no training, the cosine of the embeddings
  - local: every lab trains its own layer on its own spectra
  - centralised: one layer trained on the spectra of all labs together
  - federated (FedAvg): every lab trains on its own spectra and a server only
    averages the layer weights; no spectrum leaves the lab

DreaMS itself stays frozen. The layer is a 1024 -> 256 linear map, so it runs
on a laptop CPU in seconds. Test molecules are kept apart for all labs (split
by InChIKey), so a molecule is never used for training at one lab and for
testing at another. The settings below were fixed before looking at the test
results; they are not tuned.
"""
import argparse
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

import common as C
from exp_b_structures import molecule, top1_tanimoto

# One lab per client. crusemann_salstrep is left out because it re-analyses
# the Streptomyces data of crusemann_streptomyces (the runs are not independent).
DEFAULT_CLIENTS = [
    "duncan_salinispora", "crusemann_streptomyces", "chevrette_insect",
    "dorrestein_salbus", "tahlan_streptomyces",
]
OUT_DIR = C.EXPERIMENT_DIR / "federated_prototype"

TEST_SHARE = 0.25        # share of molecules (InChIKey) kept for testing
DIM = 256                # size of the new embedding
LR = 3e-3
WEIGHT_DECAY = 1e-4
STEPS = 200              # training steps for local and centralised
ROUNDS, LOCAL_STEPS = 20, 10   # FedAvg: 20 rounds x 10 local steps = same budget
MIN_TEST_PER_LAB = 15    # per-lab numbers only for labs with enough test spectra

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load_lab(run, in_domain_only=False):
    """Annotated spectra of one lab: embeddings, structure keys and fingerprints."""
    p = C.run_paths(run, in_domain_only)
    if not p.embeddings.exists():
        return None
    if in_domain_only and not C.read_json(p.run_info).get("spectra kept that pass DreaMS's own check"):
        return None
    meta, emb = C.load_metadata_and_embeddings(p)
    rows, keys, fps = [], [], []
    for i, smiles in enumerate(meta["smiles"]):
        mol = molecule(smiles)
        key = Chem.MolToInchiKey(mol)[:14] if mol is not None else ""
        if key:
            rows.append(i)
            keys.append(key)
            fps.append(GENERATOR.GetFingerprint(mol))
    return SimpleNamespace(run=run, emb=emb[rows], keys=np.array(keys), fps=fps)


def is_test(key):
    """The same molecule always lands on the same side, in every lab."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000 < TEST_SHARE * 1000


def tanimoto(fps):
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], dtype=np.float32)


def training_set(emb, fps):
    """Tensors for one training set. Most pairs of molecules are not similar, so
    the pairs are weighted to make every Tanimoto range (10 bins) count the same."""
    t = tanimoto(fps)
    upper = np.triu(np.ones_like(t, dtype=bool), k=1)
    bins = np.minimum((t * 10).astype(int), 9)
    counts = np.bincount(bins[upper], minlength=10).astype(float)
    w = np.where(upper, 1.0 / np.maximum(counts[bins], 1), 0.0)
    return SimpleNamespace(
        x=torch.tensor(emb), t=torch.tensor(t), w=torch.tensor(w / w.sum(), dtype=torch.float32), n=len(fps),
    )


def new_layer(seed):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(1024, DIM, bias=False)
    torch.nn.init.orthogonal_(layer.weight)  # starts close to the DreaMS cosines
    return layer


def similarity(layer, x):
    h = torch.nn.functional.normalize(layer(x), dim=1)
    return h @ h.T


def train(layer, sets, steps):
    """Full-batch training: the cosines of the new embeddings should match the Tanimoto."""
    opt = torch.optim.Adam(layer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(steps):
        opt.zero_grad()
        loss = sum((s.w * (similarity(layer, s.x) - s.t) ** 2).sum() for s in sets) / len(sets)
        loss.backward()
        opt.step()
    return layer


def fedavg(sets, seed):
    """Every round, each lab starts from the global layer, trains a few steps on
    its own spectra, and the server averages the weights (weighted by lab size)."""
    global_layer = new_layer(seed)
    share = torch.tensor([s.n for s in sets], dtype=torch.float32)
    share = share / share.sum()
    for _ in range(ROUNDS):
        weights = []
        for s in sets:
            local = new_layer(seed)
            local.load_state_dict(global_layer.state_dict())
            weights.append(train(local, [s], LOCAL_STEPS).weight.detach())
        global_layer.weight.data.copy_(sum(a * w for a, w in zip(share, weights)))
    return global_layer


def apply(layer, emb):
    with torch.no_grad():
        return similarity(layer, torch.tensor(emb)).numpy()


def scores(sim, t, keys, labs):
    """Spearman with Tanimoto (all test pairs and pairs from two different labs)
    and the analog-search top-1 Tanimoto (spectra of the same molecule skipped)."""
    iu = np.triu_indices(len(keys), k=1)
    cross = labs[iu[0]] != labs[iu[1]]
    blocked = np.eye(len(keys), dtype=bool) | (keys[:, None] == keys[None, :])
    return {
        "Spearman, all test pairs": spearmanr(t[iu], sim[iu]).correlation,
        "Spearman, pairs from two labs": spearmanr(t[iu][cross], sim[iu][cross]).correlation,
        "analog search top-1 Tanimoto": top1_tanimoto(sim, t, blocked)[0],
    }


def markdown(df):
    df = df.fillna("").astype(str)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in df.to_numpy()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", nargs="*", default=DEFAULT_CLIENTS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--in-domain-only", action="store_true",
                        help="only spectra that DreaMS's own format A check accepts")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out_dir = OUT_DIR.with_name(OUT_DIR.name + "_in_domain") if args.in_domain_only else OUT_DIR

    labs = []
    for run in args.clients:
        lab = load_lab(run, args.in_domain_only)
        if lab is None:
            print(f"{run}: no embeddings or no spectra in the DreaMS domain, skipped.")
        elif len(lab.keys) < 20:
            print(f"{run}: only {len(lab.keys)} annotated spectra, skipped.")
        else:
            labs.append(lab)
    if len(labs) < 2:
        raise SystemExit("Need at least two labs with annotated spectra.")

    # Split every lab by molecule, and build the pooled test set.
    counts, train_sets, test = [], [], {"emb": [], "fps": [], "keys": [], "lab": []}
    for lab in labs:
        test_rows = np.array([is_test(k) for k in lab.keys])
        tr = ~test_rows
        counts.append((lab.run, int(tr.sum()), int(test_rows.sum()), len(set(lab.keys[tr])), len(set(lab.keys[test_rows]))))
        train_sets.append(training_set(lab.emb[tr], [f for f, keep in zip(lab.fps, tr) if keep]))
        test["emb"].append(lab.emb[test_rows])
        test["fps"] += [f for f, keep in zip(lab.fps, test_rows) if keep]
        test["keys"] += list(lab.keys[test_rows])
        test["lab"] += [lab.run] * int(test_rows.sum())
    test_emb = np.vstack(test["emb"])
    test_t = tanimoto(test["fps"])
    test_keys, test_lab = np.array(test["keys"]), np.array(test["lab"])
    pooled_train = training_set(
        np.vstack([s.x.numpy() for s in train_sets]),
        [f for lab in labs for f, k in zip(lab.fps, lab.keys) if not is_test(k)],
    )

    def evaluate(method, sim, seed):
        rows = [{"method": method, "seed": seed, "test set": "all labs", "metric": m, "value": v}
                for m, v in scores(sim, test_t, test_keys, test_lab).items()]
        for lab in labs:  # the lab's own test spectra
            mask = test_lab == lab.run
            if mask.sum() >= MIN_TEST_PER_LAB:
                iu = np.triu_indices(int(mask.sum()), k=1)
                t, s = test_t[np.ix_(mask, mask)], sim[np.ix_(mask, mask)]
                rows.append({"method": method, "seed": seed, "test set": lab.run,
                             "metric": "Spearman, all test pairs", "value": spearmanr(t[iu], s[iu]).correlation})
        return rows

    results = evaluate("pre-trained DreaMS (no training)", test_emb @ test_emb.T, 0)
    for seed in range(args.seeds):
        print(f"Seed {seed}: local, centralised and federated training...")
        local = []
        for lab, s in zip(labs, train_sets):
            sim = apply(train(new_layer(seed), [s], STEPS), test_emb)
            local += [dict(r, method="local (mean over labs)") for r in evaluate(lab.run, sim, seed)
                      if r["test set"] == "all labs"]
            # a lab's local model on its own test spectra
            local += [dict(r, method="local (own lab only)") for r in evaluate(lab.run, sim, seed)
                      if r["test set"] == lab.run]
        results += local
        results += evaluate("centralised (all spectra in one place)", apply(train(new_layer(seed), [pooled_train], STEPS), test_emb), seed)
        # Control: all spectra in one place, but only pairs from the same lab (what FedAvg can see).
        results += evaluate("centralised, pairs within one lab only", apply(train(new_layer(seed), train_sets, STEPS), test_emb), seed)
        results += evaluate("federated (FedAvg, spectra stay in the lab)", apply(fedavg(train_sets, seed), test_emb), seed)

    df = pd.DataFrame(results)
    out_dir.mkdir(exist_ok=True)
    df.to_parquet(out_dir / "exp_c_all_runs.parquet", index=False)

    order = ["pre-trained DreaMS (no training)", "local (mean over labs)", "local (own lab only)",
             "centralised (all spectra in one place)", "centralised, pairs within one lab only",
             "federated (FedAvg, spectra stay in the lab)"]
    agg = df.groupby(["test set", "metric", "method"])["value"].agg(["mean", "std"]).reset_index()
    agg["value"] = [f"{m:.3f}" if np.isnan(s) or s == 0 else f"{m:.3f} ± {s:.3f}" for m, s in zip(agg["mean"], agg["std"])]

    main_table = agg[agg["test set"] == "all labs"].pivot(index="method", columns="metric", values="value")
    main_table = main_table.reindex([m for m in order if m in main_table.index]).reset_index()
    lab_table = agg[(agg["test set"] != "all labs")].pivot(index="method", columns="test set", values="value")
    lab_table = lab_table.reindex([m for m in order if m in lab_table.index]).reset_index()
    data_table = pd.DataFrame(counts, columns=["lab (client)", "train spectra", "test spectra", "train molecules", "test molecules"])

    text = "\n".join([
        "# Experiment C (first prototype): a federated layer on top of frozen DreaMS",
        "",
        "*Only spectra that pass DreaMS's own format A check.*" if args.in_domain_only
        else "*All spectra kept by our cleaning.*",
        "",
        f"Generated by `scripts/exp_c_federated_adapter.py`. {args.seeds} seeds (mean ± sd). "
        "Target: Tanimoto of Morgan fingerprints (radius 2, 2048 bits) of the library-annotated spectra. "
        f"Test molecules ({TEST_SHARE:.0%} of InChIKeys) are kept apart for all labs. "
        f"Layer: linear 1024 -> {DIM}; {STEPS} steps for local and centralised, "
        f"FedAvg {ROUNDS} rounds x {LOCAL_STEPS} local steps. Settings fixed in advance, not tuned.",
        "",
        "## Data per lab",
        "",
        markdown(data_table),
        "",
        "## Pooled test set (test spectra of all labs)",
        "",
        "*Pairs from two labs* uses only pairs of spectra measured in different labs. "
        "*Analog search* takes the most similar spectrum of another molecule and reports the Tanimoto of the two molecules.",
        "",
        markdown(main_table),
        "",
        "## Each lab's own test spectra (Spearman with Tanimoto)",
        "",
        f"Only labs with at least {MIN_TEST_PER_LAB} test spectra.",
        "",
        markdown(lab_table),
        "",
        "## Limitations",
        "",
        "- Library hits are putative annotations, so the target is noisy.",
        "- The DreaMS embedding model was fine-tuned on public library spectra; some of these molecules may be among them, which favours the pre-trained model.",
        "- Few annotated spectra in most labs; the per-lab numbers are uncertain.",
        "- FedAvg is simulated on one machine, without communication cost or privacy measures.",
    ])
    (out_dir / "RESULTS_federated_prototype.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
