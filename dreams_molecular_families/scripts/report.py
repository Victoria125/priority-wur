"""Step 5. Put the results of all runs side by side in RESULTS.md.

Only numbers and figures go in this file. The interpretation is written by
hand in the notebook ("My interpretation").
"""
import argparse
from datetime import date

import pandas as pd

import common as C

# Rows of the comparison table: (label in RESULTS.md, file, metric prefix).
KEY_METRICS = [
    ("A1 family recovery@5: DreaMS", "exp_a", "A1 family recovery@5: DreaMS"),
    ("A1 family recovery@5: MS2DeepScore", "exp_a", "A1 family recovery@5: MS2DeepScore"),
    ("A1 family recovery@5: binned cosine", "exp_a", "A1 family recovery@5: binned spectrum cosine"),
    ("A1 family recovery@5: precursor m/z", "exp_a", "A1 family recovery@5: precursor m/z only"),
    ("A1 family recovery@5: random", "exp_a", "A1 family recovery@5: random"),
    ("A1 DreaMS minus binned cosine (95% CI)", "exp_a", "A1 DreaMS minus binned cosine (95% CI)"),
    ("A1 DreaMS minus MS2DeepScore (95% CI)", "exp_a", "A1 DreaMS minus MS2DeepScore (95% CI)"),
    ("A2 median DreaMS sim., GNPS edges", "exp_a", "A2 median DreaMS similarity: GNPS edges"),
    ("A2 median DreaMS sim., random pairs", "exp_a", "A2 median DreaMS similarity: random pairs"),
    ("A3 largest DreaMS family (threshold 0.7)", "exp_a", "A3 largest DreaMS family"),
    ("A3 AMI, spectra in GNPS families (0.7)", "exp_a", "A3 agreement (AMI), only spectra in GNPS families"),
    ("B annotated spectra used", "exp_b", "annotated spectra used"),
    ("B unique structures", "exp_b", "unique structures"),
    ("B used for conclusions (>= 100 structures)", "exp_b", "B used for conclusions"),
    ("B1 Spearman with Tanimoto: DreaMS", "exp_b", "B1 Spearman (Tanimoto vs DreaMS)"),
    ("B1 Spearman with Tanimoto: MS2DeepScore", "exp_b", "B1 Spearman (Tanimoto vs MS2DeepScore)"),
    ("B1 Spearman with Tanimoto: modified cosine", "exp_b", "B1 Spearman (Tanimoto vs modified cosine)"),
    ("B1 Spearman with Tanimoto: binned cosine", "exp_b", "B1 Spearman (Tanimoto vs binned spectrum cosine)"),
    ("B1 DreaMS minus modified cosine (95% CI)", "exp_b", "B1 Spearman with Tanimoto, DreaMS minus modified cosine (95% CI)"),
    ("B1 DreaMS minus MS2DeepScore (95% CI)", "exp_b", "B1 Spearman with Tanimoto, DreaMS minus MS2DeepScore (95% CI)"),
    ("B1 Spearman with MHFP: DreaMS", "exp_b", "B1 Spearman (MHFP vs DreaMS)"),
    ("B1 Spearman with MHFP: MS2DeepScore", "exp_b", "B1 Spearman (MHFP vs MS2DeepScore)"),
    ("B1 Spearman with MHFP: modified cosine", "exp_b", "B1 Spearman (MHFP vs modified cosine)"),
    ("B2 analog search top-1 Tanimoto: DreaMS", "exp_b", "B2 top-1 neighbour Tanimoto, other molecules only (analog search): DreaMS"),
    ("B2 analog search top-1 Tanimoto: MS2DeepScore", "exp_b", "B2 top-1 neighbour Tanimoto, other molecules only (analog search): MS2DeepScore"),
    ("B2 analog search top-1 Tanimoto: modified cosine", "exp_b", "B2 top-1 neighbour Tanimoto, other molecules only (analog search): modified cosine"),
    ("B2 analog search top-1 Tanimoto: binned cosine", "exp_b", "B2 top-1 neighbour Tanimoto, other molecules only (analog search): binned spectrum cosine"),
    ("B2 analog search top-1 Tanimoto: random", "exp_b", "B2 top-1 neighbour Tanimoto, other molecules only (analog search): random"),
    ("B2 DreaMS minus modified cosine (95% CI)", "exp_b", "B2 analog search, DreaMS minus modified cosine (95% CI)"),
    ("B2 DreaMS minus MS2DeepScore (95% CI)", "exp_b", "B2 analog search, DreaMS minus MS2DeepScore (95% CI)"),
]

DATASET_ROWS = [
    "dataset", "MassIVE dataset (PoDP)", "GNPS task", "GNPS workflow",
    "fragment tolerance used by GNPS (Da)", "min cosine for GNPS edges", "spectra with peaks",
    "spectra (protocol)", "share (protocol)", "spectra (all_kept)", "share (all_kept)",
    "GNPS families (size >= 2) (protocol)", "GNPS edges between these spectra (protocol)",
    "library hits with a structure (SMILES) (protocol)",
]


def markdown(df: pd.DataFrame) -> str:
    """A small Markdown table without extra dependencies."""
    df = df.fillna("").astype(str)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in df.to_numpy()]
    return "\n".join(lines)


def load_run(run):
    p = C.run_paths(run)
    if not p.run_info.exists():
        return None
    out = {"paths": p, "info": C.read_json(p.run_info)}
    info = out["info"]
    for variant in C.VARIANTS:
        if info.get("spectra with peaks"):
            info[f"share ({variant})"] = f"{info.get(f'spectra ({variant})', 0) / info['spectra with peaks']:.0%}"
    for name in ["exp_a_network_sweep", "00_filter_report"]:
        path = p.results / f"{name}.parquet"
        out[name] = pd.read_parquet(path) if path.exists() else None
    for variant in C.VARIANTS:
        pv = C.run_paths(run, variant)
        for name in ["exp_a_summary", "exp_b_summary"]:
            path = pv.results / f"{name}.parquet"
            out[f"{name}_{variant}"] = pd.read_parquet(path) if path.exists() else None
    if p.embeddings_info.exists():
        out["embeddings"] = C.read_json(p.embeddings_info)
    return out


def main_table(runs, variant):
    return markdown(pd.DataFrame(
        {r: [metric(v[f"{f}_summary_{variant}"], prefix) for _, f, prefix in KEY_METRICS] for r, v in runs.items()},
        index=[label for label, _, _ in KEY_METRICS],
    ).reset_index(names=""))


def metric(summary, prefix):
    if summary is None:
        return ""
    hit = summary[summary["metric"] == prefix]
    return hit["value"].iloc[0] if len(hit) else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=list(C.RUNS))
    args = parser.parse_args()

    runs = {r: load_run(r) for r in args.runs}
    runs = {r: v for r, v in runs.items() if v is not None}
    if not runs:
        raise SystemExit("No results yet. Run the other steps first.")

    parts = [
        "# Results: DreaMS embeddings and molecular families",
        "",
        f"*Generated by `scripts/report.py` on {date.today().isoformat()}. Numbers only; "
        "the interpretation is in the notebook.*",
        "",
        "## Datasets",
        "",
        markdown(pd.DataFrame(
            {r: [v["info"].get(key, "") for key in DATASET_ROWS] for r, v in runs.items()},
            index=DATASET_ROWS,
        ).reset_index(names="")),
        "",
        "## Main numbers (DreaMS protocol)",
        "",
        "Only spectra that pass DreaMS's own format A check, given to every method exactly as GNPS "
        "gives them (`protocol` variant). Recovery@5 is the share of possible GNPS family partners "
        "among the 5 nearest neighbours (1.0 = all). Tanimoto and MHFP compare the structures of "
        "library-annotated spectra. *Analog search* skips spectra of the same molecule. Differences "
        "are shown with a 95% bootstrap interval; an interval that contains 0 means no detectable "
        "difference. Empty column: no spectra pass DreaMS's own check.",
        "",
        main_table(runs, "protocol"),
        "",
        "## Sensitivity analysis: our cleaning (all_kept)",
        "",
        "Our cleaning removes the peaks above the precursor before it applies the format A rules, so "
        "it also keeps spectra that DreaMS's own check rejects (old ion trap data in the Duncan job, "
        "noise above m/z 1000 in the Crüsemann *Streptomyces* job). Shown only to compare.",
        "",
        main_table(runs, "all_kept"),
        "",
    ]

    for run, v in runs.items():
        p = v["paths"]
        rel = p.results.relative_to(C.EXPERIMENT_DIR).as_posix()
        parts += [f"## {run}: {v['info'].get('dataset', '')}", ""]
        if "embeddings" in v:
            e = v["embeddings"]
            parts += [f"Embeddings: {e['shape'][0]:,} spectra, {e['model']}, {e['device']}, "
                      f"{e['seconds']} s ({e['computed_on'][:10]}).", ""]
        if v["00_filter_report"] is not None:
            parts += ["**Input spectra**", "", markdown(v["00_filter_report"]), ""]
        for name, title in [("exp_a_summary_protocol", "Experiment A"), ("exp_a_network_sweep", "Experiment A4: DreaMS network thresholds"),
                            ("exp_b_summary_protocol", "Experiment B")]:
            if v[name] is not None:
                parts += [f"**{title}**", "", markdown(v[name]), ""]
        for figure in ["exp_a_umap.png", "exp_a_similarity_hist.png", "exp_b_structure.png"]:
            if (p.results / figure).exists():
                parts += [f"![{figure}]({rel}/{figure})", ""]

    out = C.EXPERIMENT_DIR / "RESULTS.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
