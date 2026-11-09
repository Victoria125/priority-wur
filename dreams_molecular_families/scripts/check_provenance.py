"""Check where the spectra of every GNPS job come from.

The Paired Omics Data Platform (PoDP) links genomes to one MassIVE dataset
(RUNS[run]["massive"]). A GNPS job, however, lists its own input files in
params.xml (upload_file_mapping): files of MassIVE datasets, files uploaded
from a user's GNPS workspace, and spectral libraries used for annotation.

For every run this script:
  - separates spectra files from libraries and metadata files,
  - groups the spectra files by source (MassIVE dataset or user workspace),
  - checks that every MassIVE source is one of the documented sources
    (RUNS[run]["sources"], default: the PoDP dataset),
  - checks that the file names also exist in a documented MassIVE dataset
    (MassIVE file lists from the GNPS2 dataset cache, saved in provenance/),
and writes runs/<run>/provenance.json and provenance/RESULTS_provenance.md.
check_pipeline.py fails if a run has no provenance file or a failed check.
"""
import json
import re
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import PurePosixPath

import pandas as pd

import common as C

OUT_DIR = C.EXPERIMENT_DIR / "provenance"
DATASETTE = "https://datasetcache.gnps2.org/datasette/database/filename.json"
SPECTRA_EXT = (".mzxml", ".mzml", ".mgf", ".mzdata", ".mzdata.xml", ".cdf")
LIBRARY_PREFIXES = ("speclibs/", "staticlibs/", "CCMS_SpectralLibraries/")
MIN_NAMES_FOUND = 0.8     # share of a source's file names that must exist in a documented dataset


def massive_file_names(accession):
    """File names (without extension, lower case) of a MassIVE dataset, cached in provenance/."""
    cache = OUT_DIR / f"massive_files_{accession}.json"
    if cache.exists():
        return set(json.loads(cache.read_text()))
    params, names = {"dataset__exact": accession, "_shape": "objects", "_size": 1000}, []
    while True:
        url = f"{DATASETTE}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "priority-wur"}),
                                    timeout=120) as response:
            page = json.load(response)
        names += [r["filepath"] for r in page.get("rows", [])]
        if not page.get("next"):
            break
        params["_next"] = page["next"]
    stems = sorted({PurePosixPath(n).stem.lower() for n in names if n.lower().endswith(SPECTRA_EXT)})
    OUT_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(stems))
    return set(stems)


def job_input_files(run):
    """Input files of the GNPS job, from upload_file_mapping in params.xml."""
    with zipfile.ZipFile(C.local_archive(run)) as z:
        params = z.read(next(n for n in z.namelist() if n.endswith("params.xml"))).decode("utf-8", "ignore")
    return [e.split("|", 1)[1] for e in re.findall(r'<parameter name="upload_file_mapping">([^<]*)<', params) if "|" in e]


def source_of(path):
    m = re.match(r"(MSV\d{9})/", path)
    return m.group(1) if m else "workspace:" + path.split("/", 1)[0]


def check_run(run):
    spec = C.RUNS[run]
    declared = spec["massive"]
    documented = spec.get("sources", [declared])
    files = job_input_files(run)
    spectra = [f for f in files if f.lower().endswith(SPECTRA_EXT) and not f.startswith(LIBRARY_PREFIXES)]
    libraries = [f for f in files if f.startswith(LIBRARY_PREFIXES)]
    by_source = Counter(source_of(f) for f in spectra)
    known = {acc: massive_file_names(acc) for acc in documented}

    sources, problems = [], []
    for source, n in by_source.most_common():
        stems = {PurePosixPath(f).stem.lower() for f in spectra if source_of(f) == source}
        found = {acc: len(stems & names) for acc, names in known.items()}
        best = max(found, key=found.get)
        share = found[best] / len(stems)
        sources.append({"source": source, "spectra files": n, "names found in": best,
                        "share found": round(share, 3), "names in PoDP dataset": found.get(declared, 0)})
        if source.startswith("MSV") and source not in documented:
            problems.append(f"{source} is not a documented source")
        if share < MIN_NAMES_FOUND:
            problems.append(f"only {share:.0%} of the {source} file names are in a documented dataset")
    result = {
        "run": run, "PoDP dataset": declared, "documented sources": documented,
        "spectra files": len(spectra), "library files": len(libraries),
        "other files": len(files) - len(spectra) - len(libraries),
        "sources": sources, "problems": problems, "ok": not problems,
    }
    C.write_json(result, C.run_paths(run).provenance)
    return result


def main():
    rows = []
    for run in C.RUNS:
        r = check_run(run)
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {run}: " + "; ".join(
            f"{s['source']} ({s['spectra files']} files, {s['share found']:.0%} found in {s['names found in']})"
            for s in r["sources"]) + (f" | problems: {r['problems']}" if r["problems"] else ""))
        for s in r["sources"]:
            rows.append({"run": run, "PoDP dataset": r["PoDP dataset"], **s, "check": "ok" if r["ok"] else "FAIL"})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    table = "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                      + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()])
    (OUT_DIR / "RESULTS_provenance.md").write_text("\n".join([
        "# Where the spectra of each GNPS job come from",
        "",
        "Generated by `scripts/check_provenance.py` from `upload_file_mapping` in each GNPS job's params.xml. "
        "Spectral libraries used for annotation are not counted. *Share found*: share of the source's file "
        "names that exist in a documented MassIVE dataset (`RUNS[run]['sources']` in `scripts/common.py`).",
        "",
        table,
        "",
    ]), encoding="utf-8")


if __name__ == "__main__":
    main()
