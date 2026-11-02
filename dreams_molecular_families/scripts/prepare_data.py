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

import pandas as pd

import common as C


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

    # 1. Clean every spectrum and keep the ones that follow DreaMS format A.
    kept, reasons, seen = [], Counter(), set()
    for spec in spectra:
        if spec["scans"] in seen:
            reasons["duplicate cluster id"] += 1
            continue
        seen.add(spec["scans"])
        cleaned, problem = C.clean_and_check(spec)
        if problem:
            reasons[problem] += 1
        else:
            kept.append(cleaned)
    if not kept:
        raise SystemExit("No spectrum passed the quality rules. Check the input data.")

    # 2. One metadata row per kept spectrum, in the same order as the MGF file.
    meta = pd.DataFrame({
        "row": range(len(kept)),
        "cluster_index": [s["scans"] for s in kept],
        "precursor_mz": [s["precursor_mz"] for s in kept],
        "charge_in_gnps": [s["charge"] for s in kept],
        "n_peaks": [len(s["mz"]) for s in kept],
    })
    meta = meta.merge(nodes, on="cluster_index", how="left")
    meta["gnps_family"] = meta["gnps_family"].fillna(-1).astype(int)
    meta = meta.merge(library, on="cluster_index", how="left")

    # Family size counted among the kept spectra only. A GNPS family that keeps
    # only one spectrum after cleaning behaves like a singleton in our analysis.
    in_family = meta["gnps_family"] >= 0
    sizes = meta.loc[in_family, "gnps_family"].map(meta.loc[in_family, "gnps_family"].value_counts())
    meta["family_size"] = 1
    meta.loc[in_family, "family_size"] = sizes.astype(int)
    meta = meta.sort_values("row").reset_index(drop=True)

    # 3. Keep only network edges between kept spectra.
    kept_ids = set(meta["cluster_index"])
    edges = edges[edges["id1"].isin(kept_ids) & edges["id2"].isin(kept_ids)]

    # 4. Write everything.
    C.write_mgf(kept, p.clean_mgf)
    meta.to_parquet(p.metadata, index=False)
    edges.to_parquet(p.edges, index=False)

    report = pd.DataFrame(
        [("read from GNPS", len(spectra))]
        + [(f"removed: {reason}", n) for reason, n in reasons.most_common()]
        + [("kept for DreaMS", len(kept))],
        columns=["step", "spectra"],
    )
    report.to_parquet(p.results / "00_filter_report.parquet", index=False)

    spec = C.RUNS.get(args.run, {})
    n_families = meta.loc[meta["family_size"] >= 2, "gnps_family"].nunique()
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
        "spectra kept": len(meta),
        "spectra in a GNPS family (size >= 2)": int((meta["family_size"] >= 2).sum()),
        "GNPS families (size >= 2)": int(n_families),
        "GNPS edges between kept spectra": len(edges),
        "spectra with a GNPS library hit": int(meta["compound_name"].notna().sum()),
        "library hits with a structure (SMILES)": int(meta["smiles"].notna().sum()),
        "clean MGF md5": C.md5sum(p.clean_mgf),
    }
    C.write_json(info, p.run_info)

    print(report.to_string(index=False))
    print()
    for key, value in info.items():
        print(f"{key}: {value}")
    print(f"\nWrote {p.clean_mgf.name}, {p.metadata.name} and {p.edges.name} in {p.data}")


if __name__ == "__main__":
    main()
