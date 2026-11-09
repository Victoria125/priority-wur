"""Step 1. Get a GNPS molecular networking job, clean the spectra and
write the input files for DreaMS.

Usage:
  python scripts/prepare_data.py --run duncan_salinispora           # a dataset listed in common.RUNS
  python scripts/prepare_data.py --run my_run --task-id <ID>        # another GNPS classic job
  python scripts/prepare_data.py --run my_run --zip <file.zip>      # a GNPS zip you downloaded yourself
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dreams.utils.dformats import DataFormatA

import common as C

DREAMS_FORMAT_A = DataFormatA()


def passes_dreams_check(spec) -> bool:
    """DreaMS's own format A check, on the spectrum as GNPS gives it.

    Our cleaning first removes the peaks above the precursor, so it keeps some
    spectra that DreaMS itself would reject (for example old ion trap data with
    noise up to m/z 2000). This flag tells them apart."""
    peaks = np.vstack([spec["mz"], spec["intensity"]])
    charge = spec["charge"] or None  # 0 = unknown, DreaMS then skips the charge check
    with np.errstate(divide="ignore"):
        return DREAMS_FORMAT_A.val_spec(peaks, spec["precursor_mz"], charge=charge) is True


def get_archive(p, task_id: str | None, zip_path: str | None) -> Path:
    if zip_path:
        return Path(zip_path)

    # 1. The copy that scripts/download_data.py keeps in data/raw/gnps/.
    local = C.local_archive(p.name)
    if local is not None and C.is_gnps_archive(local):
        print(f"Using the local GNPS job {local.relative_to(C.REPO_DIR)}")
        return local

    # 2. A pinned copy on GitHub (only for the Duncan job).
    spec = C.RUNS.get(p.name, {})
    dest = p.data / "gnps_job.zip"
    if "fallback_url" in spec:
        if not dest.exists() or C.md5sum(dest) != spec["fallback_md5"]:
            print(f"Downloading the GNPS job for {p.name} from GitHub...")
            C.download(spec["fallback_url"], dest)
        if C.md5sum(dest) != spec["fallback_md5"]:
            raise SystemExit(f"The download is corrupted. Delete {dest} and run again.")
        return dest

    # 3. GNPS itself.
    task_id = task_id or spec.get("task_id")
    if task_id is None:
        raise SystemExit(f"Unknown run '{p.name}'. Give --task-id or --zip.")
    if dest.exists() and C.is_gnps_archive(dest):
        return dest
    for url in C.GNPS_URLS:
        print("Trying", url.format(task_id))
        try:
            C.download(url.format(task_id), dest, post=True)
            if C.is_gnps_archive(dest):
                return dest
        except Exception as error:  # GNPS can be slow or offline
            print("  failed:", error)
    raise SystemExit(
        "Could not download this GNPS job. Download the zip by hand from the GNPS job page "
        "and run: python scripts/prepare_data.py --run <name> --zip <file.zip>"
    )


def check_gnps_job(spectra, nodes, edges, library) -> int:
    """Stop if the files of the GNPS job do not agree with each other.

    In every job checked so far, the spectra with peaks are exactly the nodes of
    the network, and edges and library hits only point to nodes. If that is not
    true, the counts in RESULTS.md would be wrong, so we stop here."""
    with_peaks = {s["scans"] for s in spectra if len(s["mz"]) > 0}
    node_ids = set(nodes["cluster_index"])
    problems = []
    if with_peaks != node_ids:
        problems.append(f"{len(with_peaks - node_ids):,} spectra with peaks are not network nodes and "
                        f"{len(node_ids - with_peaks):,} nodes have no spectrum with peaks")
    bad_edges = int((~edges["id1"].isin(node_ids) | ~edges["id2"].isin(node_ids)).sum())
    if bad_edges:
        problems.append(f"{bad_edges:,} network edges point to unknown nodes")
    bad_hits = int((~library["cluster_index"].isin(node_ids)).sum())
    if bad_hits:
        problems.append(f"{bad_hits:,} library hits point to unknown nodes")
    if problems:
        raise SystemExit("The files of this GNPS job do not agree:\n  - " + "\n  - ".join(problems))
    return len(node_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--zip", default=None)
    args = parser.parse_args()

    p = C.run_paths(args.run)
    p.data.mkdir(parents=True, exist_ok=True)
    p.results.mkdir(parents=True, exist_ok=True)

    archive = get_archive(p, args.task_id, args.zip)
    paths = C.extract_gnps_archive(archive, p.gnps)
    params = C.read_gnps_params(paths.get("params"))

    spectra = C.read_mgf(paths["spectra"])
    nodes = C.read_nodes(paths["nodes"])
    edges = C.read_edges(paths["edges"])
    library = C.read_library_hits(paths.get("library"))
    print(f"Read {len(spectra):,} spectra, {len(edges):,} network edges, "
          f"{len(library):,} library hits.")
    n_nodes = check_gnps_job(spectra, nodes, edges, library)
    print(f"Checked: the {n_nodes:,} spectra with peaks are the network nodes, "
          "and edges and library hits point to them.")

    # 1. Two versions of the input (see common.VARIANTS):
    #    all_kept: our cleaning, then the DreaMS format A rules;
    #    protocol: the spectra that pass DreaMS's own check, unchanged.
    kept, in_domain, protocol, reasons, seen = [], [], [], Counter(), set()
    for spec in spectra:
        if spec["scans"] in seen:
            reasons["duplicate cluster id"] += 1
            continue
        seen.add(spec["scans"])
        dreams_ok = len(spec["mz"]) > 0 and passes_dreams_check(spec)
        if dreams_ok:
            protocol.append(spec)
        cleaned, problem = C.clean_and_check(spec)
        if problem:
            reasons[problem] += 1
        else:
            kept.append(cleaned)
            in_domain.append(dreams_ok)
    if not kept and not protocol:
        raise SystemExit("No spectrum passed the quality rules. Check the input data.")

    # 2. One metadata table per version, in the same order as its MGF file.
    #    All network edges are saved; the experiments keep the ones they need.
    edges.to_parquet(p.edges, index=False)
    info_variants = {}
    for variant, chosen in [("all_kept", kept), ("protocol", protocol)]:
        pv = C.run_paths(args.run, variant)
        if not chosen:
            for old in (pv.clean_mgf, pv.metadata):
                old.unlink(missing_ok=True)
            print(f"No spectra for the {variant} variant.")
            info_variants[variant] = {"spectra": 0}
            continue
        meta = spectra_table(chosen, nodes, library)
        if variant == "all_kept":
            meta["in_dreams_domain"] = in_domain
        C.write_mgf(chosen, pv.clean_mgf)
        meta.to_parquet(pv.metadata, index=False)
        ids = set(meta["cluster_index"])
        info_variants[variant] = {
            "spectra": len(meta),
            "spectra in a GNPS family (size >= 2)": int((meta["family_size"] >= 2).sum()),
            "GNPS families (size >= 2)": int(meta.loc[meta["family_size"] >= 2, "gnps_family"].nunique()),
            "GNPS edges between these spectra": int((edges["id1"].isin(ids) & edges["id2"].isin(ids)).sum()),
            "library hits with a structure (SMILES)": int(meta["smiles"].notna().sum()),
            "MGF md5": C.md5sum(pv.clean_mgf),
        }

    report = pd.DataFrame(
        [("read from GNPS", len(spectra))]
        + [(f"removed: {reason}", n) for reason, n in reasons.most_common()]
        + [("kept by our cleaning (all_kept)", len(kept)),
           ("pass DreaMS's own check, unchanged (protocol)", len(protocol))],
        columns=["step", "spectra"],
    )
    report.to_parquet(p.results / "00_filter_report.parquet", index=False)

    spec = C.RUNS.get(args.run, {})
    info = {
        "run": args.run,
        "dataset": spec.get("label", ""),
        "MassIVE dataset (PoDP)": spec.get("massive", ""),
        "GNPS task": params.get("task", args.task_id or spec.get("task_id", "")),
        "GNPS workflow": params.get("workflow", "unknown"),
        "description": params.get("desc", ""),
        "MassIVE datasets in the GNPS job": params.get("reanalyzed_datasets", ""),
        "fragment tolerance used by GNPS (Da)": params.get("tolerance.Ion_tolerance", ""),
        "min cosine for GNPS edges": params.get("PAIRS_MIN_COSINE", ""),
        "min matched peaks for GNPS edges": params.get("MIN_MATCHED_PEAKS", ""),
        "spectra read": len(spectra),
        # Some GNPS jobs list every input scan, most of them without peaks.
        # The spectra with peaks are the real input (the network nodes).
        "spectra with peaks": n_nodes,
        "GNPS network edges": len(edges),
    }
    for variant, values in info_variants.items():
        for key, value in values.items():
            info[f"{key} ({variant})"] = value
    C.write_json(info, p.run_info)

    print(report.to_string(index=False))
    print()
    for key, value in info.items():
        print(f"{key}: {value}")


def spectra_table(spectra, nodes, library) -> pd.DataFrame:
    """One row per spectrum, in MGF order, with its GNPS family and library hit."""
    meta = pd.DataFrame({
        "row": range(len(spectra)),
        "cluster_index": [s["scans"] for s in spectra],
        "precursor_mz": [s["precursor_mz"] for s in spectra],
        "charge_in_gnps": [s["charge"] for s in spectra],
        "n_peaks": [len(s["mz"]) for s in spectra],
    })
    meta = meta.merge(nodes, on="cluster_index", how="left")
    meta["gnps_family"] = meta["gnps_family"].fillna(-1).astype(int)
    meta = meta.merge(library, on="cluster_index", how="left")
    # Family size counted among these spectra only (see common.add_family_size).
    return C.add_family_size(meta).sort_values("row").reset_index(drop=True)


if __name__ == "__main__":
    main()
