"""Step 6 (Experiment C). A small layer on frozen DreaMS embeddings, trained
centrally or across clients: does any training help, and does federated
training keep up with central training?

Data: library-annotated spectra that pass the checks of annotations.py, one
spectrum per molecule and lab. The training target is the structural
similarity (Morgan Tanimoto) of two compounds, as in MS2DeepScore. The layer
is a 1024 -> 256 linear map on top of the frozen DreaMS embeddings.

Robust design (fixed before looking at test results):
  - 5 random splits by molecule (first InChIKey block): 25% test, 15% of the
    rest validation, the rest training. A molecule is never in two parts.
  - Three ways to split the training compounds into clients:
      labs       the natural split, one client per lab (labs with >= 5 compounds)
      iid        the pooled compounds shuffled into 5 equal clients
      dirichlet  5 clients with Dirichlet(0.5) shares of 8 chemical classes
                 (k-means on the fingerprints): a label-skew partition
  - Methods, all with the same model, learning rate and budget (300 steps, or
    30 rounds x 10 local steps), and the step or round chosen on the
    validation compounds (never on the test compounds):
      pre-trained DreaMS (no training), local (mean over clients),
      centralised (all compounds in one place), FedAvg, FedProx (mu = 0.01),
      SplitFedV2-like split learning (the server trains on one client after
      the other; Thapa et al. 2022, Tirana et al. 2026).
  - Test: Spearman between the new similarity and Tanimoto over all pairs of
    test compounds, and the analog-search top-1 Tanimoto.
  - Every split gives one value per method; we report the mean with a 95%
    t-interval over splits, and paired differences against pre-trained DreaMS
    and against centralised training. An advantage is claimed only when the
    interval of the paired difference excludes 0.
"""
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

import common as C
from annotations import validated_annotations
from exp_b_structures import top1_tanimoto

DEFAULT_LABS = ["duncan_salinispora", "crusemann_salstrep", "chevrette_insect",
                "dorrestein_salbus", "tahlan_streptomyces"]
OUT_DIR = C.EXPERIMENT_DIR / "federated_prototype"

N_SPLITS = 5
TEST_SHARE, VAL_SHARE = 0.25, 0.15
N_CLIENTS = 5                  # for the iid and dirichlet partitions
DIRICHLET_ALPHA, N_CLASSES = 0.5, 8
MIN_LAB_COMPOUNDS = 5          # labs with fewer training compounds are not clients
DIM, LR, WEIGHT_DECAY = 256, 3e-3, 1e-4
STEPS, ROUNDS, LOCAL_STEPS = 300, 30, 10
EVAL_EVERY = 10                # validation every 10 steps (= every round)
FEDPROX_MU = 0.01
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_compounds(labs, variant):
    """One row per (lab, molecule): embedding, structure key and fingerprint."""
    rows = []
    for run in labs:
        p = C.run_paths(run, variant)
        if not (p.metadata.exists() and p.embeddings.exists()):
            print(f"{run}: no {variant} spectra or embeddings, skipped.")
            continue
        meta, emb = C.load_metadata_and_embeddings(p)
        selected, _ = validated_annotations(meta)
        for _, s in selected.iterrows():
            rows.append({"lab": run, "key": s["key"], "emb": emb[s["row"]], "fp": GENERATOR.GetFingerprint(s["mol"])})
    return pd.DataFrame(rows)


def split_molecules(keys, split):
    """Random split by molecule: 'test', 'val' or 'train' for every row."""
    unique = np.array(sorted(set(keys)))
    rng = np.random.default_rng(1000 + split)
    rng.shuffle(unique)
    n_test = round(TEST_SHARE * len(unique))
    n_val = round(VAL_SHARE * (len(unique) - n_test))
    part = {k: "test" for k in unique[:n_test]}
    part.update({k: "val" for k in unique[n_test:n_test + n_val]})
    part.update({k: "train" for k in unique[n_test + n_val:]})
    return np.array([part[k] for k in keys])


def tanimoto(fps):
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps], dtype=np.float32)


def training_set(df):
    """Tensors for one client. Pairs are weighted so that every Tanimoto range
    (10 bins) counts the same, because most pairs of molecules are not similar."""
    t = tanimoto(list(df["fp"]))
    upper = np.triu(np.ones_like(t, dtype=bool), k=1)
    bins = np.minimum((t * 10).astype(int), 9)
    counts = np.bincount(bins[upper], minlength=10).astype(float)
    w = np.where(upper, 1.0 / np.maximum(counts[bins], 1), 0.0)
    return SimpleNamespace(x=torch.tensor(np.stack(df["emb"])), t=torch.tensor(t),
                           w=torch.tensor(w / max(w.sum(), 1e-12), dtype=torch.float32), n=len(df))


def partitions(train, split):
    """{partition name: list of client DataFrames} for the training compounds."""
    rng = np.random.default_rng(2000 + split)
    out = {}
    labs = [g for _, g in train.groupby("lab") if len(g) >= MIN_LAB_COMPOUNDS]
    if len(labs) >= 2:
        out["labs"] = labs
    order = rng.permutation(len(train))
    out["iid"] = [train.iloc[order[i::N_CLIENTS]] for i in range(N_CLIENTS)]
    fps = np.array([list(fp) for fp in train["fp"]], dtype=np.float32)
    classes = KMeans(n_clusters=N_CLASSES, n_init=10, random_state=split).fit_predict(fps)
    client = np.empty(len(train), dtype=int)
    for c in range(N_CLASSES):
        members = np.where(classes == c)[0]
        shares = rng.dirichlet([DIRICHLET_ALPHA] * N_CLIENTS)
        client[members] = rng.choice(N_CLIENTS, size=len(members), p=shares)
    groups = [train.iloc[np.where(client == k)[0]] for k in range(N_CLIENTS)]
    out["dirichlet"] = [g for g in groups if len(g) >= 2]
    return out


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
def new_layer(seed):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(1024, DIM, bias=False)
    torch.nn.init.orthogonal_(layer.weight)
    return layer


def similarity(layer, x):
    h = torch.nn.functional.normalize(layer(x), dim=1)
    return h @ h.T


def loss_of(layer, s):
    return (s.w * (similarity(layer, s.x) - s.t) ** 2).sum()


class Selector:
    """Keeps the weights with the best validation Spearman."""

    def __init__(self, val):
        self.x = torch.tensor(np.stack(val["emb"]))
        self.t = tanimoto(list(val["fp"]))
        self.iu = np.triu_indices(len(val), k=1)
        self.best, self.weights = -np.inf, None

    def __call__(self, layer):
        with torch.no_grad():
            s = similarity(layer, self.x).numpy()
        rho = spearmanr(self.t[self.iu], s[self.iu]).correlation
        if rho > self.best:
            self.best, self.weights = rho, layer.weight.detach().clone()

    def best_layer(self, seed):
        layer = new_layer(seed)
        layer.weight.data.copy_(self.weights)
        return layer


def train_central(sets, val, seed):
    """Full-batch training on the given client sets together (one optimiser)."""
    layer, select = new_layer(seed), Selector(val)
    opt = torch.optim.Adam(layer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        (sum(loss_of(layer, s) for s in sets) / len(sets)).backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            select(layer)
    return select.best_layer(seed)


def train_federated(sets, val, seed, mu=0.0):
    """FedAvg (mu = 0) or FedProx (mu > 0): each client starts from the global
    layer, trains LOCAL_STEPS steps on its own compounds, the server averages."""
    global_layer, select = new_layer(seed), Selector(val)
    share = torch.tensor([s.n for s in sets], dtype=torch.float32)
    share = share / share.sum()
    for _ in range(ROUNDS):
        weights = []
        start = global_layer.weight.detach().clone()
        for s in sets:
            local = new_layer(seed)
            local.weight.data.copy_(start)
            opt = torch.optim.Adam(local.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            for _ in range(LOCAL_STEPS):
                opt.zero_grad()
                loss = loss_of(local, s)
                if mu:
                    loss = loss + mu / 2 * ((local.weight - start) ** 2).sum()
                loss.backward()
                opt.step()
            weights.append(local.weight.detach())
        global_layer.weight.data.copy_(sum(a * w for a, w in zip(share, weights)))
        select(global_layer)
    return select.best_layer(seed)


def train_split_sequential(sets, val, seed):
    """SplitFedV2-like: the server part is trained on one client after the other."""
    layer, select = new_layer(seed), Selector(val)
    opt = torch.optim.Adam(layer.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(ROUNDS):
        for s in sets:
            for _ in range(LOCAL_STEPS):
                opt.zero_grad()
                loss_of(layer, s).backward()
                opt.step()
        select(layer)
    return select.best_layer(seed)


def test_scores(layer, test):
    x = np.stack(test["emb"])
    if layer is None:
        s = x @ x.T
    else:
        with torch.no_grad():
            s = similarity(layer, torch.tensor(x)).numpy()
    t = tanimoto(list(test["fp"]))
    iu = np.triu_indices(len(test), k=1)
    keys = test["key"].to_numpy()
    blocked = np.eye(len(test), dtype=bool) | (keys[:, None] == keys[None, :])
    return {"Spearman with Tanimoto": spearmanr(t[iu], s[iu]).correlation,
            "analog search top-1 Tanimoto": top1_tanimoto(s, t, blocked)[0]}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def t_interval(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return f"{v.mean():.3f}" if len(v) else ""
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():+.3f} [{v.mean() - half:+.3f}, {v.mean() + half:+.3f}]"


def markdown(df):
    df = df.fillna("").astype(str)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in df.to_numpy()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labs", nargs="*", default=DEFAULT_LABS)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--variant", default="protocol", choices=list(C.VARIANTS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    out_dir = OUT_DIR if args.variant == "protocol" else OUT_DIR.with_name(f"{OUT_DIR.name}_{args.variant}")

    data = load_compounds(args.labs, args.variant)
    if data["key"].nunique() < 50:
        raise SystemExit("Fewer than 50 annotated compounds that pass the checks: too few for this experiment.")

    results, sizes = [], []
    for split in range(args.splits):
        part = split_molecules(data["key"].to_numpy(), split)
        train, val, test = data[part == "train"], data[part == "val"], data[part == "test"]
        seed = split
        base = test_scores(None, test)
        for partition, clients in partitions(train, split).items():
            sets = [training_set(c) for c in clients]
            sizes.append({"split": split, "partition": partition, "clients": len(clients),
                          "compounds per client": ", ".join(str(len(c)) for c in clients),
                          "validation": len(val), "test": len(test)})
            methods = {"pre-trained DreaMS (no training)": base}
            local = [test_scores(train_central([s], val, seed), test) for s in sets]
            methods["local (mean over clients)"] = {m: np.mean([r[m] for r in local]) for m in base}
            methods["centralised (all compounds in one place)"] = test_scores(
                train_central([training_set(pd.concat(clients))], val, seed), test)
            methods["FedAvg"] = test_scores(train_federated(sets, val, seed), test)
            methods[f"FedProx (mu={FEDPROX_MU})"] = test_scores(train_federated(sets, val, seed, FEDPROX_MU), test)
            methods["SplitFedV2-like (clients one after the other)"] = test_scores(
                train_split_sequential(sets, val, seed), test)
            for method, scores in methods.items():
                for metric, value in scores.items():
                    results.append({"split": split, "partition": partition, "method": method,
                                    "metric": metric, "value": value})
        print(f"split {split + 1}/{args.splits} done")

    df = pd.DataFrame(results)
    out_dir.mkdir(exist_ok=True)
    df.to_parquet(out_dir / "exp_c_all_runs.parquet", index=False)
    pd.DataFrame(sizes).to_parquet(out_dir / "exp_c_partitions.parquet", index=False)

    tables = []
    for metric in ["Spearman with Tanimoto", "analog search top-1 Tanimoto"]:
        d = df[df["metric"] == metric]
        wide = d.pivot_table(index=["partition", "split"], columns="method", values="value")
        rows = []
        for partition, g in wide.groupby(level=0):
            for method in wide.columns:
                rows.append({
                    "partition": partition, "method": method,
                    f"{metric} (mean, 95% CI over splits)": t_interval(g[method]),
                    "minus pre-trained (paired)": "" if method.startswith("pre-trained") else
                    t_interval(g[method] - g["pre-trained DreaMS (no training)"]),
                    "minus centralised (paired)": "" if method.startswith(("centralised", "pre-trained")) else
                    t_interval(g[method] - g["centralised (all compounds in one place)"]),
                })
        tables.append((metric, pd.DataFrame(rows)))

    size_table = (pd.DataFrame(sizes).groupby("partition")
                  .agg(clients=("clients", "first"), example_split_0=("compounds per client", "first"),
                       validation=("validation", "first"), test=("test", "first")).reset_index())
    per_lab = data.groupby("lab")["key"].nunique().rename("compounds").reset_index()
    text = "\n".join([
        "# Experiment C: a layer on frozen DreaMS, trained centrally and across clients",
        "",
        f"Generated by `scripts/exp_c_federated_adapter.py`, variant `{args.variant}`. {args.splits} random splits by "
        f"molecule (test {TEST_SHARE:.0%}, validation {VAL_SHARE:.0%} of the rest). The design is described at the top "
        "of the script; it was fixed before looking at test results. Values are means over splits with a 95% "
        "t-interval. *Paired* columns compare methods on the same split. **An advantage is claimed only when the "
        "paired interval excludes 0.**",
        "",
        "## Compounds",
        "",
        "Library-annotated compounds that pass the checks of `annotations.py`, per lab:",
        "",
        markdown(per_lab),
        "",
        "Clients per partition (compounds per client shown for split 0):",
        "",
        markdown(size_table),
        "",
        *[part for metric, table in tables for part in (f"## {metric}", "", markdown(table), "")],
        "## Limitations",
        "",
        "- Few compounds: most come from one lab (Tahlan), so the natural partition has few, very unequal clients.",
        "- Library annotations are putative even after the checks; the target is noisy.",
        "- The DreaMS embedding model was fine-tuned on public library spectra; some test molecules may be among "
        "them, which favours the pre-trained model.",
        "- Simulated on one machine, without communication costs or privacy measures.",
    ])
    (out_dir / "RESULTS_federated_prototype.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
