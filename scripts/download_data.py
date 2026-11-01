
"""Download the actinobacterial paired-omics data found by podp_inventory.py.

Steps (run any subset; default is genomes + gnps + spectra + extras):
    genomes  resolve every linked genome to an NCBI assembly, then download
             its FASTA and GenBank files. PoDP often gives only JGI IDs, so
             genomes are matched by BioSample or strain name.
    gnps     the GNPS molecular-network results registered in PoDP (clustered
             MS/MS spectra + network): small, and what NPLinker reads
    spectra  the raw mzXML/mzML files referenced by PoDP genome-metabolome links
    extras   MIBiG 4.0, small processed GEO tables, lydicamycin genomes
    massive  every file of one MassIVE dataset collection
             (e.g. --accession MSV000094769 --collection ccms_peak)

Everything lands in data/raw/ (git-ignored), and every attempt is appended
to data/derived/download_manifest.csv. Downloads resume: finished files are
skipped, and partial files (*.part) are retried. A disk guard stops the run
before free space drops below --min-free-gb.

Downloaded files are data: run this with `python -I`, from the project root.

Usage:
    python -I scripts/download_data.py --dry-run
    python -I scripts/download_data.py genomes spectra extras
    python -I scripts/download_data.py massive --accession MSV000094769 --collection ccms_peak
"""
import argparse
import csv
import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
MANIFEST = DERIVED / "download_manifest.csv"

MASSIVE_DL = "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile"
DATASETTE = "https://datasetcache.gnps2.org/datasette/database/filename.json"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GNPS_DL = "https://gnps.ucsd.edu/ProteoSAFe/DownloadResult"  # needs POST
GNPS_VIEWS = ("download_cytoscape_data", "download_clustered_spectra")
UA = {"User-Agent": "priority-wur-download/0.1"}

MIBIG = [
    "https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz",
    "https://dl.secondarymetabolites.org/mibig/mibig_gbk_4.0.tar.gz",
]
GEO_SERIES = ["GSE318666", "GSE132487"]
EXTRA_BIOPROJECTS = ["PRJNA985726"]  # lydicamycin Streptomyces genomes (DTU)

ASSEMBLY_RE = re.compile(r"GC[AF]_\d+\.\d+")
LEVEL_RANK = {"Complete Genome": 3, "Chromosome": 2, "Scaffold": 1, "Contig": 0}


class DiskFull(Exception):
    pass


class PermanentError(Exception):
    """A failure that retrying will not fix."""


# --------------------------------------------------------------------- http

def open_url(url, timeout=300, method="GET"):
    data = b"" if method == "POST" else None
    req = urllib.request.Request(url, data=data, headers=UA, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url, retries=4):
    for attempt in range(retries):
        try:
            with open_url(url, 120) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def get_text(url):
    with open_url(url, 120) as resp:
        return resp.read().decode("utf-8", "replace")


def check_disk(min_free_gb):
    free = shutil.disk_usage(RAW).free / 1e9
    if free < min_free_gb:
        raise DiskFull(f"stopping: {free:.0f} GB free, guard is {min_free_gb} GB")


TRANSIENT_HTTP = {429, 502, 503, 504}


def backoff(exc, attempt):
    """Seconds to wait before retrying, or None to give up now.

    Rate limits and gateway errors wait (Retry-After when given); network
    errors back off exponentially; other HTTP errors (404, or the 500 MassIVE
    returns for a missing path) are permanent and fail fast.
    """
    if isinstance(exc, PermanentError):
        return None
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code not in TRANSIENT_HTTP:
            return None
        retry_after = exc.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return int(retry_after)
    return min(300, 15 * 2 ** attempt)


def download(url, dest, expected=None, min_free_gb=40, retries=8, method="GET"):
    """Fetch url to dest via a .part file. Returns (status, bytes)."""
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        return "skipped", dest.stat().st_size
    check_disk(min_free_gb)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    error = None
    for attempt in range(retries):
        try:
            with open_url(url, method=method) as resp, part.open("wb") as fh:
                shutil.copyfileobj(resp, fh, 1 << 20)
            size = part.stat().st_size
            if expected and size != expected:
                raise IOError(f"got {size} bytes, expected {expected}")
            with part.open("rb") as fh:
                head = fh.read(200).lstrip().lower()
            if head.startswith((b"<!doctype html", b"<html")):
                raise PermanentError("server returned an HTML page instead of the file")
            part.replace(dest)
            return "ok", size
        except Exception as exc:
            error = exc
            wait = backoff(exc, attempt)
            if wait is None:
                break
            time.sleep(wait)
    part.unlink(missing_ok=True)
    return f"failed: {error}", 0


class Manifest:
    fields = ["time", "step", "source", "url", "path", "bytes", "status"]

    def __init__(self):
        self.rows, self.lock = [], threading.Lock()

    def add(self, step, source, url, path, size, status):
        with self.lock:
            self.rows.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step,
                "source": source, "url": url,
                "path": str(Path(path).relative_to(ROOT)) if path else "",
                "bytes": size, "status": status,
            })

    def write(self):
        new = not MANIFEST.exists()
        with MANIFEST.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fields)
            if new:
                writer.writeheader()
            writer.writerows(self.rows)
        self.rows = []


def run_parallel(jobs, workers, manifest, step, min_free_gb):
    """jobs: list of (source, url, dest, expected[, method]). Stops cleanly on DiskFull."""
    stop = threading.Event()
    done, total_bytes = 0, 0

    def work(job):
        if stop.is_set():
            return job, "not started (disk guard)", 0
        url, dest, expected = job[1:4]
        method = job[4] if len(job) > 4 else "GET"
        try:
            return job, *download(url, dest, expected, min_free_gb, method=method)
        except DiskFull as exc:
            stop.set()
            return job, f"not started ({exc})", 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, j) for j in jobs]):
            job, status, size = fut.result()
            source, url, dest = job[:3]
            manifest.add(step, source, url, dest, size, status)
            done += 1
            total_bytes += size
            if done % 25 == 0 or done == len(jobs):
                print(f"[{step}] {done}/{len(jobs)} files, {total_bytes / 1e9:.1f} GB", flush=True)
    if stop.is_set():
        print(f"[{step}] disk guard reached - remaining files were not downloaded", flush=True)
    return not stop.is_set()


# ------------------------------------------------------------------ massive

def list_massive(accession):
    params = {"dataset__exact": accession, "_shape": "objects", "_size": 1000}
    while True:
        page = get_json(f"{DATASETTE}?{urllib.parse.urlencode(params)}")
        yield from page.get("rows", [])
        if not page.get("next"):
            return
        params["_next"] = page["next"]


def massive_url(path):
    return f"{MASSIVE_DL}?" + urllib.parse.urlencode({"file": f"f.{path}", "forceDownload": "true"})


def massive_sizes(accession):
    by_path, by_name = {}, defaultdict(set)
    for row in list_massive(accession):
        rel = row["filepath"].lower()
        by_path[rel] = row.get("size")
        by_name[PurePosixPath(rel).name].add(row.get("size"))
    return by_path, by_name


# ------------------------------------------------------------- PoDP inputs

def selected_projects(accessions):
    with (DERIVED / "podp_projects.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if accessions:
        return {r["podp_id"] for r in rows if r["metabolomics_id"] in accessions}
    return {r["podp_id"] for r in rows if float(r["actino_fraction"]) >= 0.5}


def selected_links(podp_ids):
    with (DERIVED / "podp_links.csv").open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["podp_id"] in podp_ids]


# --------------------------------------------------------------------- GNPS

def gnps_jobs(podp_ids):
    """The GNPS molecular-network results each PoDP project registers.

    Clustered MS/MS spectra + network files: what NPLinker reads, and orders
    of magnitude smaller than the raw files (MSV000078839: 35 MB vs 148 GB).
    """
    jobs = []
    for pid in sorted(podp_ids):
        record = json.loads((RAW / "podp" / f"{pid}.json").read_text(encoding="utf-8"))
        meta = record.get("project", record).get("metabolomics", {}).get("project", {})
        task = str(meta.get("molecular_network") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", task):
            continue
        acc = meta.get("GNPSMassIVE_ID") or pid
        for view in GNPS_VIEWS:
            url = f"{GNPS_DL}?" + urllib.parse.urlencode({"task": task, "view": view})
            jobs.append((acc, url, RAW / "gnps" / acc / task / f"{view}.zip", None, "POST"))
    return jobs


# ------------------------------------------------------------------ spectra

def spectra_jobs(links):
    jobs, sizes = [], {}
    for link in links:
        parsed = urllib.parse.urlparse(link["metabolomics_file"].strip())
        path = re.sub(r"/{2,}", "/", urllib.parse.unquote(parsed.path)).lstrip("/")
        if parsed.netloc.endswith("massive.ucsd.edu"):
            url = massive_url(path)
        elif parsed.scheme in ("http", "https"):
            url = link["metabolomics_file"].strip()
        else:
            continue
        acc = path.split("/", 1)[0]
        if acc not in sizes and acc.startswith("MSV"):
            sizes[acc] = massive_sizes(acc)
        expected = None
        if acc in sizes and "/" in path:
            by_path, by_name = sizes[acc]
            rel = path.split("/", 1)[1].lower()
            names = by_name.get(PurePosixPath(rel).name, set())
            expected = by_path.get(rel) or (next(iter(names)) if len(names) == 1 else None)
        jobs.append((link["metabolomics_id"], url, RAW / "spectra" / path, expected))
    unique = {j[2]: j for j in jobs}
    return list(unique.values())


def order_small_first(jobs):
    """Whole datasets, smallest first, so a disk stop leaves complete sets."""
    totals = defaultdict(int)
    for source, _, _, expected in jobs:
        totals[source] += expected or 0
    return sorted(jobs, key=lambda j: (totals[j[0]], j[0], str(j[2])))


# ------------------------------------------------------------------ genomes

def strain_key(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def label_strain(label):
    """'Streptomyces sp. CNB091' -> 'CNB091', 'X sp. ATCC 39727' -> 'ATCC39727'.

    Keeps tokens after the first that carry digits, plus an upper-case
    culture-collection prefix (ATCC, DSM, NRRL...) written apart from its number.
    """
    tokens = (label or "").split()[1:]
    keep = [t for i, t in enumerate(tokens)
            if re.search(r"\d", t)
            or (re.fullmatch(r"[A-Z]{2,}", t) and i + 1 < len(tokens)
                and re.match(r"\d", tokens[i + 1]))]
    return strain_key(" ".join(keep))


def label_species(label, genus):
    """Species epithet following the genus ('' for 'sp.' or when absent)."""
    low = [t.lower() for t in (label or "").split()]
    if genus and genus.lower() in low:
        i = low.index(genus.lower())
        if i + 1 < len(low) and re.fullmatch(r"[a-z]+", low[i + 1]) and low[i + 1] != "sp":
            return low[i + 1]
    return ""


def genus_index(genus):
    """All NCBI assemblies of a genus, indexed by BioSample and strain key."""
    by_biosample, by_strain = {}, defaultdict(list)
    params = {"page_size": 1000}
    while True:
        url = f"{DATASETS}/genome/taxon/{urllib.parse.quote(genus)}/dataset_report?" \
              + urllib.parse.urlencode(params)
        page = get_json(url)
        for rep in page.get("reports", []):
            org = rep.get("organism", {})
            info = rep.get("assembly_info", {})
            entry = (rep.get("accession", ""), info.get("assembly_level", ""),
                     org.get("organism_name", ""))
            bs = (info.get("biosample") or {}).get("accession")
            if bs:
                by_biosample.setdefault(bs, []).append(entry)
            strain = (org.get("infraspecific_names") or {}).get("strain")
            for key in {strain_key(strain), label_strain(org.get("organism_name", ""))} - {""}:
                by_strain[key].append(entry)
        token = page.get("next_page_token")
        if not token:
            return by_biosample, by_strain
        params["page_token"] = token
        time.sleep(0.4)


def best(entries):
    """Prefer RefSeq, then the most complete assembly level."""
    return max(entries, key=lambda e: (e[0].startswith("GCF"), LEVEL_RANK.get(e[1], -1), e[0]))


def eutils_assembly(biosample=None, nuccore=None):
    time.sleep(0.4)
    if nuccore:
        data = get_json(f"{EUTILS}/elink.fcgi?dbfrom=nuccore&db=assembly&retmode=json"
                        f"&id={urllib.parse.quote(nuccore)}")
        ids = [i for ls in data.get("linksets", []) for db in ls.get("linksetdbs", [])
               for i in db.get("links", [])]
    else:
        term = urllib.parse.quote(f"{biosample}[BioSample]")
        ids = get_json(f"{EUTILS}/esearch.fcgi?db=assembly&retmode=json&term={term}")[
            "esearchresult"]["idlist"]
    if not ids:
        return ""
    time.sleep(0.4)
    summ = get_json(f"{EUTILS}/esummary.fcgi?db=assembly&retmode=json&id={','.join(ids[:20])}")
    accs = [summ["result"][i]["assemblyaccession"] for i in summ["result"].get("uids", [])]
    return min(accs, key=lambda a: (not a.startswith("GCF"), a)) if accs else ""


def resolve_genomes(links):
    genomes = {}
    for link in links:
        key = (link["genome_label"], link["genome_accession"])
        genomes.setdefault(key, {
            "metabolomics_id": link["metabolomics_id"], "genome_label": link["genome_label"],
            "genus": link["genus"], "podp_id_type": link["genome_id_type"],
            "podp_accession": link["genome_accession"], "biosample": link["biosample"],
        })
    rows = list(genomes.values())

    need_index = {r["genus"] for r in rows
                  if r["genus"] and not ASSEMBLY_RE.fullmatch(r["podp_accession"])}
    indexes = {}
    for genus in sorted(need_index):
        print(f"[genomes] indexing NCBI assemblies for {genus}", flush=True)
        try:
            indexes[genus] = genus_index(genus)
        except Exception as exc:
            print(f"[genomes] could not index {genus}: {exc}", flush=True)

    for r in rows:
        acc, method = resolve_one(r, indexes.get(r["genus"]))
        r["resolved_assembly"], r["method"] = acc, method
    return rows


def resolve_one(r, index):
    """(assembly accession, method) for one PoDP genome; method says how sure we are."""
    if ASSEMBLY_RE.fullmatch(r["podp_accession"]):
        return r["podp_accession"], "podp_assembly"
    if index and r["biosample"] in index[0]:
        return best(index[0][r["biosample"]])[0], "biosample_match"
    entries = index[1].get(label_strain(r["genome_label"]), []) if index else []
    if entries:
        species = label_species(r["genome_label"], r["genus"])
        same = [e for e in entries if species in e[2].lower()] if species else entries
        if same:
            return best(same)[0], "strain_match"
        # One organism behind every hit (often a GCA/GCF pair): a renamed
        # species, e.g. the 2020 split of Salinispora pacifica. Keep, but flag.
        if len({e[2] for e in entries}) == 1:
            return best(entries)[0], "strain_match_species_renamed"
    try:
        if r["biosample"].startswith("SAM"):
            acc = eutils_assembly(biosample=r["biosample"])
            if acc:
                return acc, "biosample_eutils"
        if r["podp_id_type"] in ("GenBank", "RefSeq") and r["podp_accession"]:
            acc = eutils_assembly(nuccore=r["podp_accession"])
            if acc:
                return acc, "nuccore_eutils"
    except Exception:
        pass
    return "", "unresolved"


def bioproject_assemblies(bioproject):
    page = get_json(f"{DATASETS}/genome/bioproject/{bioproject}/dataset_report?page_size=1000")
    return sorted({rep["accession"] for rep in page.get("reports", [])})


def genome_url(acc):
    return (f"{DATASETS}/genome/accession/{acc}/download?"
            "include_annotation_type=GENOME_FASTA&include_annotation_type=GENOME_GBFF")


def fetch_genomes(accessions, manifest, step, workers, min_free_gb):
    jobs = []
    for acc in sorted(set(accessions)):
        dest_dir = RAW / "genomes" / acc
        if any(dest_dir.glob("*.fna")):
            manifest.add(step, acc, genome_url(acc), dest_dir, 0, "skipped")
            continue
        jobs.append((acc, genome_url(acc), RAW / "genomes" / "_zips" / f"{acc}.zip", None))
    ok = run_parallel(jobs, min(workers, 3), manifest, step, min_free_gb)
    for acc, _, zip_path, _ in jobs:
        if not zip_path.exists():
            continue
        dest_dir = RAW / "genomes" / acc
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.namelist():
                    name = PurePosixPath(member).name  # flatten: no path traversal
                    if name.endswith((".fna", ".gbff")):
                        with zf.open(member) as src, (dest_dir / name).open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            zip_path.unlink()
        except zipfile.BadZipFile:
            manifest.add(step, acc, genome_url(acc), zip_path, 0, "failed: bad zip")
    return ok


# ------------------------------------------------------------------- extras

def geo_jobs(series):
    folder = f"{series[:-3]}nnn"
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{folder}/{series}/suppl/"
    names = re.findall(r'href="([^"?/][^"]*)"', get_text(base))
    return [(series, base + n, RAW / "geo" / series / urllib.parse.unquote(n), None)
            for n in names if not n.endswith("/")]


# --------------------------------------------------------------------- main

def write_genome_table(rows):
    path = DERIVED / "genome_accessions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[genomes] wrote {path.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="*", default=["genomes", "gnps", "spectra", "extras"],
                        help="genomes, gnps, spectra, extras, massive")
    parser.add_argument("--accessions", nargs="*", help="limit PoDP steps to these MassIVE IDs")
    parser.add_argument("--accession", help="massive step: dataset to mirror")
    parser.add_argument("--collection", help="massive step: only this collection")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel downloads; MassIVE answers 429 above ~3")
    parser.add_argument("--min-free-gb", type=float, default=40)
    parser.add_argument("--dry-run", action="store_true", help="estimate sizes only")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = Manifest()

    podp_ids = selected_projects(args.accessions)
    links = selected_links(podp_ids)
    print(f"{len(podp_ids)} PoDP projects, {len(links)} genome-metabolome links selected", flush=True)

    try:
        # Genomes first: small and the most reusable, so a disk stop never costs them.
        if "genomes" in args.steps and not args.dry_run:
            rows = resolve_genomes(links)
            write_genome_table(rows)
            counts = defaultdict(int)
            for r in rows:
                counts[r["method"]] += 1
            print(f"[genomes] {len(rows)} genomes: {dict(counts)}", flush=True)
            fetch_genomes([r["resolved_assembly"] for r in rows if r["resolved_assembly"]],
                          manifest, "genomes", args.workers, args.min_free_gb)

        if "gnps" in args.steps and not args.dry_run:
            run_parallel(gnps_jobs(podp_ids), 2, manifest, "gnps", args.min_free_gb)

        if "spectra" in args.steps or args.dry_run:
            jobs = order_small_first(spectra_jobs(links))
            known = sum(j[3] or 0 for j in jobs)
            unknown = sum(1 for j in jobs if not j[3])
            print(f"[spectra] {len(jobs)} unique files, {known / 1e9:.1f} GB known size"
                  f" ({unknown} files of unknown size)", flush=True)
            per_set = defaultdict(int)
            for j in jobs:
                per_set[j[0]] += j[3] or 0
            for acc, size in sorted(per_set.items(), key=lambda kv: kv[1]):
                print(f"    {acc}: {size / 1e9:.2f} GB", flush=True)
            if not args.dry_run:
                run_parallel(jobs, args.workers, manifest, "spectra", args.min_free_gb)

        if "extras" in args.steps and not args.dry_run:
            jobs = [("MIBiG 4.0", u, RAW / "mibig" / u.rsplit("/", 1)[1], None) for u in MIBIG]
            for series in GEO_SERIES:
                try:
                    jobs += geo_jobs(series)
                except Exception as exc:
                    manifest.add("extras", series, "", "", 0, f"failed listing: {exc}")
            run_parallel(jobs, args.workers, manifest, "extras", args.min_free_gb)
            accs = []
            for bp in EXTRA_BIOPROJECTS:
                try:
                    accs += bioproject_assemblies(bp)
                except Exception as exc:
                    manifest.add("extras", bp, "", "", 0, f"failed: {exc}")
            fetch_genomes(accs, manifest, "extras-genomes", args.workers, args.min_free_gb)

        if "massive" in args.steps:
            rows = [r for r in list_massive(args.accession)
                    if not args.collection or r.get("collection") == args.collection]
            jobs = [(args.accession, massive_url(f"{args.accession}/{r['filepath']}"),
                     RAW / "massive" / args.accession / r["filepath"], r.get("size"))
                    for r in rows]
            print(f"[massive] {args.accession}: {len(jobs)} files, "
                  f"{sum(j[3] or 0 for j in jobs) / 1e9:.1f} GB", flush=True)
            if not args.dry_run:
                run_parallel(jobs, args.workers, manifest, "massive", args.min_free_gb)
    finally:
        if manifest.rows:
            manifest.write()
            print(f"Manifest: {MANIFEST.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
