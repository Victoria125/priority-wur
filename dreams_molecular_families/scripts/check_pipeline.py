"""Automatic checks of the pipeline. Run after any change; exit code 0 = all passed.

Missing data or results are failures, not skips. The only allowed absence is
a variant that run_info.json declares empty (0 spectra), and then its files
must not exist either.

1. Unit checks: the GNPS consistency check, the bootstrap interval, the
   annotation checks, top_peaks and the spectra fingerprint.
2. Required files: the DreaMS tutorial reproduction (and that it reproduced),
   and for every run its run_info, input report and provenance (which must
   have passed), and for every non-empty variant the MGF, metadata, DreaMS and
   MS2DeepScore embeddings (up to date, one per spectrum, the same 100 input
   peaks) and the summaries of experiments A and B. Protocol spectra must all
   pass DreaMS's own format A check. The overall result files must exist.
3. Regression: the summaries of experiments A and B equal the saved reference
   (tests/reference/). After a change that should alter results on purpose,
   look at the differences and save a new reference with --save-reference.

Usage:
  python scripts/check_pipeline.py
  python scripts/check_pipeline.py --save-reference
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from annotations import ANNOTATION_QC_VERSION, normalise_adduct, validated_annotations
from prepare_data import check_gnps_job, passes_dreams_check

REFERENCE_DIR = C.EXPERIMENT_DIR / "tests" / "reference"
SUMMARIES = ["exp_a_summary.parquet", "exp_b_summary.parquet"]
OVERALL_RESULTS = [
    "RESULTS.md",
    "federated_prototype/RESULTS_federated_prototype.md",
    "lab_shift/RESULTS_lab_shift_protocol.md",
    "linking_feasibility/RESULTS_linking_feasibility.md",
    "provenance/RESULTS_provenance.md",
]

failures = []


def check(condition, message):
    print(("ok    " if condition else "FAIL  ") + message)
    if not condition:
        failures.append(message)
    return condition


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

    # Annotations: caffeine [M+H]+ = 195.0877. Rows: a good exact hit, the same molecule
    # as a sodium adduct (duplicate), an analog hit, a low-score hit and a wrong-mass [M+H]+ hit.
    caffeine, ethanol = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CCO"
    meta = pd.DataFrame({
        "smiles": [caffeine, caffeine, ethanol, ethanol, caffeine],
        "adduct": ["[M+H]+", "M+Na", "M+H", "M+H", "M+H"],
        "mass_diff": [0.001, 0.002, 12.5, 0.001, 0.001],
        "library_score": [0.9, 0.95, 0.9, 0.4, 0.9],
        "shared_peaks": [10, 10, 10, 10, 10],
        "library_mz": [195.0877, 217.0696, 47.049, 47.049, 300.0],
        "precursor_mz": [195.0877, 217.0696, 59.5, 47.049, 300.0],
        "inchi": [None] * 5,
    })
    selected, _ = validated_annotations(meta)
    check(len(selected) == 1 and selected["row"].iloc[0] == 0,
          "annotation checks drop analog, low-score and wrong-mass hits and keep one spectrum per molecule")
    check(normalise_adduct("[M+H]+") == normalise_adduct("M+H") == normalise_adduct("[M+H]") == "M+H",
          "adduct names are normalised")

    spec = {"mz": np.arange(150, dtype=float), "intensity": np.arange(150, dtype=float) + 1}
    top = C.top_peaks(spec)
    check(len(top["mz"]) == C.INPUT_PEAKS and top["mz"].min() == 50 and np.all(np.diff(top["mz"]) > 0),
          f"top_peaks keeps the {C.INPUT_PEAKS} most intense peaks in m/z order")

    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.mgf", Path(tmp) / "b.mgf"
        s = {"scans": 7, "precursor_mz": 300.1, "mz": np.array([100.0, 200.0]), "intensity": np.array([5.0, 9.0])}
        C.write_mgf([dict(s, charge=1)], a)
        C.write_mgf([dict(s, charge=0)], b)
        check(C.spectra_fingerprint(a) == C.spectra_fingerprint(b) and C.md5sum(a) != C.md5sum(b),
              "the spectra fingerprint ignores the charge line, the file md5 does not")
        check(C.read_mgf(b)[0]["charge"] == 0 and C.read_mgf(a)[0]["charge"] == 1,
              "unknown charge stays unknown when written and read back")


def file_checks():
    reproduction = C.EXPERIMENT_DIR / "reproduction" / "dreams_networking_tutorial.json"
    if check(reproduction.exists(), "DreaMS tutorial reproduction exists"):
        check(C.read_json(reproduction).get("reproduced") is True, "DreaMS tutorial was reproduced")

    for run in C.RUNS:
        p = C.run_paths(run)
        if not check(p.run_info.exists(), f"{run}: run_info.json exists"):
            continue
        info = C.read_json(p.run_info)
        check(info.get("annotation_qc_version") == ANNOTATION_QC_VERSION,
              f"{run}: run_info matches the current chemical annotation QC version")
        check(p.input_report.exists(), f"{run}: input_report.parquet exists")
        if check(p.provenance.exists(), f"{run}: provenance.json exists"):
            check(C.read_json(p.provenance).get("ok") is True, f"{run}: provenance check passed")
        for variant in C.VARIANTS:
            pv = C.run_paths(run, variant)
            label = f"{run} ({variant})"
            n = info.get(f"spectra ({variant})")
            if not check(n is not None, f"{label}: spectra count recorded in run_info"):
                continue
            if n == 0:
                check(not pv.clean_mgf.exists() and not pv.metadata.exists(),
                      f"{label}: declared empty and has no files")
                continue
            if not (check(pv.clean_mgf.exists(), f"{label}: MGF exists")
                    and check(pv.metadata.exists(), f"{label}: metadata exists")):
                continue
            meta = pd.read_parquet(pv.metadata)
            spectra = C.read_mgf(pv.clean_mgf)
            check(len(meta) == len(spectra) == n, f"{label}: metadata, MGF and run_info agree ({n} spectra)")
            check(meta["row"].tolist() == list(range(n)), f"{label}: row indices match the spectral file")
            check(meta["cluster_index"].is_unique, f"{label}: spectrum identifiers are unique")
            for name, emb_path, info_path in [("DreaMS", pv.embeddings, pv.embeddings_info),
                                              ("MS2DeepScore", pv.ms2ds_embeddings, pv.ms2ds_info)]:
                if not check(emb_path.exists() and info_path.exists(), f"{label}: {name} embeddings exist"):
                    continue
                check(np.load(emb_path, mmap_mode="r").shape[0] == n, f"{label}: one {name} embedding per spectrum")
                expected = ({"model": "DreaMS_embedding (pre-trained, no fine-tuning)",
                             "dreams_commit": C.DREAMS_COMMIT} if name == "DreaMS"
                            else {"model": "MS2DeepScore pre-trained (Zenodo 17826815)",
                                  "input_peaks": C.INPUT_PEAKS})
                check(C.embeddings_up_to_date(info_path, pv.clean_mgf, expected=expected),
                      f"{label}: {name} embeddings match spectra and model identity")
            if pv.ms2ds_info.exists():
                check(C.read_json(pv.ms2ds_info).get("input_peaks") == C.INPUT_PEAKS,
                      f"{label}: MS2DeepScore got the same {C.INPUT_PEAKS} peaks as DreaMS")
            for summary in SUMMARIES:
                summary_path = pv.results / summary
                if check(summary_path.exists(), f"{label}: {summary} exists"):
                    check(summary_path.stat().st_mtime >= pv.metadata.stat().st_mtime,
                          f"{label}: {summary} was regenerated after metadata")
            if variant == "protocol":
                check(all(passes_dreams_check(s) for s in spectra),
                      f"{label}: every spectrum passes DreaMS's own check")

    for relative in OVERALL_RESULTS:
        check((C.EXPERIMENT_DIR / relative).exists(), f"{relative} exists")


def regression_checks(save):
    for run in C.RUNS:
        for variant in C.VARIANTS:
            results = C.run_paths(run, variant).results
            reference = REFERENCE_DIR / run / variant
            for name in SUMMARIES:
                current = results / name
                if not current.exists():
                    continue  # already reported by file_checks if it was required
                if save:
                    reference.mkdir(parents=True, exist_ok=True)
                    shutil.copy(current, reference / name)
                    continue
                if not check((reference / name).exists(), f"{run} ({variant}) {name}: reference exists"):
                    continue
                before, after = pd.read_parquet(reference / name), pd.read_parquet(current)
                same = check(before.equals(after), f"{run} ({variant}) {name} equals the reference")
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
