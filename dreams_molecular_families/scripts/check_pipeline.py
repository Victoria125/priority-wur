"""Automatic checks of the pipeline. Run after any change to the scripts.

1. Unit checks: the GNPS consistency check catches broken inputs, and the
   bootstrap interval behaves as expected.
2. File checks, for every run and variant: metadata, MGF and embeddings have
   the same number of spectra, the embeddings belong to this exact MGF file
   (md5), and every protocol spectrum passes DreaMS's own check.
3. Regression check: the experiment summaries are the same as in the saved
   reference (tests/reference/). After a change that should alter results on
   purpose, look at the differences and save a new reference with
   --save-reference.

Exit code 0 = all checks passed.

Usage:
  python scripts/check_pipeline.py
  python scripts/check_pipeline.py --save-reference
"""
import argparse
import shutil
import sys

import numpy as np
import pandas as pd

import common as C
from prepare_data import check_gnps_job, passes_dreams_check

REFERENCE_DIR = C.EXPERIMENT_DIR / "tests" / "reference"
SUMMARIES = ["exp_a_summary.parquet", "exp_b_summary.parquet"]

failures = []


def check(condition, message):
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)


def unit_checks():
    spectra = [{"scans": 1, "mz": np.array([100.0])}, {"scans": 2, "mz": np.array([150.0])},
               {"scans": 3, "mz": np.array([])}]
    nodes = pd.DataFrame({"cluster_index": [1, 2], "gnps_family": [0, 0]})
    edges = pd.DataFrame({"id1": [1], "id2": [2], "gnps_cosine": [0.9]})
    library = pd.DataFrame({"cluster_index": [2]})
    check(check_gnps_job(spectra, nodes, edges, library) == 2, "GNPS check accepts consistent files")
    broken = {
        "an extra node": (spectra, pd.concat([nodes, pd.DataFrame({"cluster_index": [9], "gnps_family": [1]})]), edges, library),
        "an edge to an unknown node": (spectra, nodes, pd.DataFrame({"id1": [1], "id2": [7], "gnps_cosine": [0.8]}), library),
        "a library hit on an unknown node": (spectra, nodes, edges, pd.DataFrame({"cluster_index": [5]})),
    }
    for name, args in broken.items():
        try:
            check_gnps_job(*args)
            check(False, f"GNPS check catches {name}")
        except SystemExit:
            check(True, f"GNPS check catches {name}")

    mean, low, high = C.bootstrap_mean_ci(np.full(50, 0.3))
    check(abs(mean - 0.3) < 1e-12 and abs(low - 0.3) < 1e-12 and abs(high - 0.3) < 1e-12,
          "bootstrap interval of a constant is the constant")
    values = np.r_[np.zeros(50), np.ones(50)]
    _, low_iid, high_iid = C.bootstrap_mean_ci(values)
    _, low_grp, high_grp = C.bootstrap_mean_ci(values, groups=np.repeat([0, 1], 50))
    check(high_grp - low_grp > high_iid - low_iid, "resampling whole groups gives a wider interval")


def file_checks():
    for run in C.RUNS:
        for variant in C.VARIANTS:
            p = C.run_paths(run, variant)
            label = f"{run} ({variant})"
            if not p.metadata.exists():
                print(f"skip  {label}: no spectra for this variant")
                continue
            meta = pd.read_parquet(p.metadata)
            spectra = C.read_mgf(p.clean_mgf)
            check(len(meta) == len(spectra), f"{label}: metadata and MGF have the same number of spectra")
            md5 = C.md5sum(p.clean_mgf)
            for name, emb_path, info_path in [("DreaMS", p.embeddings, p.embeddings_info),
                                              ("MS2DeepScore", p.ms2ds_embeddings, p.ms2ds_info)]:
                if not emb_path.exists():
                    print(f"skip  {label}: no {name} embeddings")
                    continue
                check(np.load(emb_path, mmap_mode="r").shape[0] == len(meta),
                      f"{label}: one {name} embedding per spectrum")
                check(C.read_json(info_path).get("clean_mgf_md5") == md5,
                      f"{label}: {name} embeddings belong to this MGF file")
            if variant == "protocol":
                check(all(passes_dreams_check(dict(s, charge=1)) for s in spectra),
                      f"{label}: every spectrum passes DreaMS's own check")


def regression_checks(save):
    for run in C.RUNS:
        for variant in C.VARIANTS:
            results = C.run_paths(run, variant).results
            reference = REFERENCE_DIR / run / variant
            for name in SUMMARIES:
                current = results / name
                if not current.exists():
                    continue
                if save:
                    reference.mkdir(parents=True, exist_ok=True)
                    shutil.copy(current, reference / name)
                    continue
                if not (reference / name).exists():
                    print(f"skip  {run} ({variant}) {name}: no reference saved yet")
                    continue
                before, after = pd.read_parquet(reference / name), pd.read_parquet(current)
                same = before.equals(after)
                check(same, f"{run} ({variant}) {name} equals the reference")
                if not same:
                    merged = before.merge(after, on="metric", how="outer", suffixes=(" reference", " now"))
                    changed = merged[merged["value reference"] != merged["value now"]]
                    print(changed[["metric", "value reference", "value now"]].to_string(index=False))
    if save:
        print(f"Saved the current summaries as reference in {REFERENCE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-reference", action="store_true")
    args = parser.parse_args()
    unit_checks()
    file_checks()
    regression_checks(args.save_reference)
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
