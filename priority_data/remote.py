"""On-demand access to the public datasets, local copy first, remote otherwise.

Nothing here builds a new dataset: every function reads the original
public records (MassIVE/GNPS2, NCBI) at the moment they are needed.
When a file was already fetched by scripts/download_data.py it is read
from data/raw/; otherwise it is fetched from the source, parsed, and
discarded, so datasets too large to keep (e.g. MSV000094769, 161 GB) can
still be used from code without storing them.

    from priority_data import remote
    remote.massive_files("MSV000094769")[:3]
    for spec in remote.iter_spectra("MSV000094769", "ccms_peak/peak/MSV000088196_P2R1C2.mzML"):
        ...
    remote.usi_spectrum("mzspec:MSV000078839:CNB091_A1_M:scan:100")
    remote.genome_fasta("GCF_000010605.1")
    run = remote.rnaseq_runs("PRJNA1397920")[0]
    for name, seq, qual in remote.iter_fastq(run["fastq_urls"][0], limit=1000):
        ...
"""
import gzip
import json
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from pyteomics import mgf, mzml, mzxml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

MASSIVE_DL = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile"
DATASETTE = "https://datasetcache.gnps2.org/datasette/database/filename.json"
USI_API = "https://metabolomics-usi.gnps2.org/json/"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2"
ENA = "https://www.ebi.ac.uk/ena/portal/api/filereport"
UA = {"User-Agent": "priority-wur-remote/0.1"}


def _open(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def _spool(url, retries=6):
    """Remote file -> anonymous temporary file (deleted on close), retrying on 429/503."""
    for attempt in range(retries):
        tmp = tempfile.TemporaryFile()
        try:
            with _open(url) as resp:
                shutil.copyfileobj(resp, tmp, 1 << 20)
            tmp.seek(0)
            return tmp
        except urllib.error.HTTPError as exc:
            tmp.close()
            if exc.code not in (429, 503) or attempt == retries - 1:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            time.sleep(int(retry_after) if retry_after.isdigit() else min(300, 15 * 2 ** attempt))


def _json(url, retries=3):
    for attempt in range(retries):
        try:
            with _open(url, 120) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


# ---------------------------------------------------------------- MassIVE

def massive_files(accession, collection=None, suffix=None):
    """File list of a MassIVE dataset (path, collection, size) from the GNPS2 cache."""
    params = {"dataset__exact": accession, "_shape": "objects", "_size": 1000}
    rows = []
    while True:
        page = _json(f"{DATASETTE}?{urllib.parse.urlencode(params)}")
        rows += page.get("rows", [])
        if not page.get("next"):
            break
        params["_next"] = page["next"]
    return [
        {"path": r["filepath"], "collection": r.get("collection"), "size": r.get("size")}
        for r in rows
        if (collection is None or r.get("collection") == collection)
        and (suffix is None or r["filepath"].lower().endswith(suffix.lower()))
    ]


def massive_url(accession, path):
    return f"{MASSIVE_DL}?" + urllib.parse.urlencode(
        {"file": f"f.{accession}/{path}", "forceDownload": "true"})


def local_path(accession, path):
    """Where download_data.py would have stored this file, if it did."""
    for base in (RAW / "spectra" / accession, RAW / "massive" / accession):
        candidate = base / path
        if candidate.exists():
            return candidate
    return None


def _normalise(spec, is_mzml):
    if is_mzml:
        level = spec.get("ms level")
        match = re.search(r"scan=(\d+)", spec.get("id", ""))
        scan = int(match.group(1)) if match else spec.get("index", 0) + 1
        rt = spec.get("scanList", {}).get("scan", [{}])[0].get("scan start time")
        precursor = None
        if level and level > 1:
            ions = (spec.get("precursorList", {}).get("precursor", [{}])[0]
                    .get("selectedIonList", {}).get("selectedIon", [{}]))
            precursor = ions[0].get("selected ion m/z")
    else:
        level = spec.get("msLevel")
        scan = int(spec.get("num", 0))
        rt = spec.get("retentionTime")
        precursor = (spec.get("precursorMz") or [{}])[0].get("precursorMz")
    return {
        "scan": scan, "ms_level": level,
        "rt": float(rt) if rt is not None else None,
        "precursor_mz": float(precursor) if precursor is not None else None,
        "mz": spec.get("m/z array"), "intensity": spec.get("intensity array"),
    }


def iter_spectra(accession, path, ms_level=2):
    """Yield spectra of one mzML/mzXML file: local copy if present, else remote.

    pyteomics needs a seekable file, so a remote file is spooled into a
    temporary file that is deleted as soon as parsing ends; nothing is kept.
    Each spectrum is a dict: scan, ms_level, rt, precursor_mz, mz, intensity.
    """
    is_mzml = path.lower().endswith(".mzml")
    reader = mzml.read if is_mzml else mzxml.read
    local = local_path(accession, path)
    source = local.open("rb") if local else _spool(massive_url(accession, path))
    with source:
        for spec in reader(source, use_index=False, read_schema=False):
            item = _normalise(spec, is_mzml)
            if ms_level is None or item["ms_level"] == ms_level:
                yield item


def export_ms2_mgf(accession, paths, out_path):
    """Keep only the MS/MS spectra of (remote) files, as one compact MGF.

    This is how to use a dataset too large to store: each raw file is
    fetched, reduced to its fragmentation spectra and dropped. Returns the
    number of spectra written.
    """
    def spectra():
        for path in paths:
            stem = PurePosixPath(path).stem
            for s in iter_spectra(accession, path, ms_level=2):
                yield {
                    "m/z array": s["mz"], "intensity array": s["intensity"],
                    "params": {
                        "title": f"{accession}:{stem}:scan:{s['scan']}",
                        "pepmass": s["precursor_mz"], "rtinseconds": s["rt"],
                        "usi": f"mzspec:{accession}:{stem}:scan:{s['scan']}",
                    },
                }

    count = 0

    def counted():
        nonlocal count
        for item in spectra():
            count += 1
            yield item

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mgf.write(counted(), output=str(out_path))
    return count


def usi_spectrum(usi):
    """One spectrum by Universal Spectrum Identifier, via the GNPS2 USI service."""
    data = _json(f"{USI_API}?{urllib.parse.urlencode({'usi1': usi})}")
    return {"usi": usi, "precursor_mz": data.get("precursor_mz"), "peaks": data.get("peaks", [])}


# --------------------------------------------------------------- RNA-seq

def rnaseq_runs(accession):
    """Sequencing runs of a BioProject/study with FASTQ links and sizes (ENA portal API)."""
    params = {"accession": accession, "result": "read_run", "format": "json",
              "fields": "run_accession,sample_title,fastq_ftp,fastq_bytes,read_count"}
    runs = _json(f"{ENA}?{urllib.parse.urlencode(params)}")
    for run in runs:
        run["fastq_urls"] = [f"https://{u}" for u in run.get("fastq_ftp", "").split(";") if u]
        run["fastq_bytes"] = sum(int(b) for b in str(run.get("fastq_bytes", "")).split(";") if b)
    return runs


def iter_fastq(url, limit=None):
    """Stream (name, sequence, quality) reads from a remote .fastq.gz; nothing is saved."""
    with _open(url) as resp, gzip.open(resp, "rt") as fh:
        for n, header in enumerate(iter(fh.readline, "")):
            if limit is not None and n >= limit:
                return
            seq, _, qual = fh.readline().rstrip(), fh.readline(), fh.readline().rstrip()
            yield header[1:].rstrip(), seq, qual


# ------------------------------------------------------------------- NCBI

def genome_fasta(accession):
    """Path to a genome FASTA: from data/raw/genomes, else fetched once into data/cache."""
    for base in (RAW / "genomes" / accession, CACHE / "genomes" / accession):
        hits = sorted(base.glob("*.fna"))
        if hits:
            return hits[0]
    dest = CACHE / "genomes" / accession
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "download.zip"
    url = f"{DATASETS}/genome/accession/{accession}/download?include_annotation_type=GENOME_FASTA"
    with _open(url) as resp, zip_path.open("wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = PurePosixPath(member).name
            if name.endswith(".fna"):
                with zf.open(member) as src, (dest / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    zip_path.unlink()
    hits = sorted(dest.glob("*.fna"))
    if not hits:
        raise FileNotFoundError(f"NCBI returned no FASTA for {accession}")
    return hits[0]
