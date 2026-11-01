"""Summarise the files of a MassIVE/GNPS dataset without downloading it.

Lists the files through the GNPS2 dataset cache (a Datasette instance) and
reports what you need to know before calling a dataset "paired":
counts per collection and extension, total size, which earlier MassIVE
datasets the files were copied from, genome accessions, duplicated files,
and candidate metadata / sample-mapping files.

Standard library only.

Usage:
    python scripts/massive_files.py MSV000094769
    python scripts/massive_files.py MSV000094769 --out data/derived/MSV000094769_files.csv
"""
import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import PurePosixPath

DATASETTE = "https://datasetcache.gnps2.org/datasette/database/filename.json"
METADATA_EXT = {".tsv", ".csv", ".txt", ".xlsx", ".xls", ".json"}


def get_json(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "priority-wur-inventory/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def list_files(accession):
    # Follow the `next` token, not `next_url`: the server builds next_url
    # from an internal hostname that does not resolve from outside.
    params = {"dataset__exact": accession, "_shape": "objects", "_size": 1000}
    while True:
        page = get_json(f"{DATASETTE}?{urllib.parse.urlencode(params)}")
        yield from page.get("rows", [])
        if not page.get("next"):
            return
        params["_next"] = page["next"]


def extension(path):
    suffixes = PurePosixPath(path.lower()).suffixes
    if suffixes[-1:] == [".gz"] and len(suffixes) > 1:
        return "".join(suffixes[-2:])
    return suffixes[-1] if suffixes else "(none)"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("accession", help="MassIVE accession, e.g. MSV000094769")
    parser.add_argument("--out", help="optional CSV with one row per file")
    args = parser.parse_args()

    files = list(list_files(args.accession))
    if not files:
        print(f"No files found for {args.accession} in the GNPS2 dataset cache.")
        return

    by_collection = Counter(f.get("collection") or "(none)" for f in files)
    size_gb = defaultdict(float)
    for f in files:
        size_gb[f.get("collection") or "(none)"] += (f.get("size") or 0) / 1e9
    by_ext = Counter(extension(f["filepath"]) for f in files)

    names = [PurePosixPath(f["filepath"]).name for f in files]
    sources = Counter(m.group(1) for n in names if (m := re.match(r"(MSV\d{9})_", n)))
    genomes = sorted({m.group(0) for n in names if (m := re.search(r"GC[AF]_\d+\.\d+", n))})
    dupes = sum(1 for n, c in Counter(names).items() if c > 1)
    metadata = [f["filepath"] for f in files
                if extension(f["filepath"]) in METADATA_EXT or "metadata" in f["filepath"].lower()]

    print(f"{args.accession}: {len(files)} files, {sum(size_gb.values()):.1f} GB total")
    print("\nBy collection:")
    for col, n in by_collection.most_common():
        print(f"  {col:<20}{n:>6} files {size_gb[col]:>8.1f} GB")
    print("\nBy extension:")
    for ext, n in by_ext.most_common():
        print(f"  {ext:<20}{n:>6}")
    print(f"\nFile names shared by more than one collection: {dupes}")
    if sources:
        print("\nFiles copied from earlier MassIVE datasets (by file-name prefix):")
        for acc, n in sources.most_common():
            print(f"  {acc}: {n}")
    print(f"\nGenome accessions in file names: {len(genomes)}"
          + (f" (e.g. {', '.join(genomes[:3])})" if genomes else ""))
    print(f"\nCandidate metadata / sample-mapping files: {len(metadata)}")
    for path in metadata[:20]:
        print(f"  {path}")
    if not metadata:
        print("  none - the genome <-> spectrum pairing is NOT documented in the deposit")

    if args.out:
        fields = ["filepath", "collection", "size", "create_time"]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(files)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
