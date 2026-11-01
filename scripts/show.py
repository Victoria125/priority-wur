"""Short tables for showing the results on screen.

Reads the tables in data/derived/ (run podp_inventory.py and massive_files.py
first). Standard library only.

Usage:
    python scripts/show.py datasets   # actinobacterial PoDP datasets
    python scripts/show.py links      # known BGC <-> MS2 links vs distinct BGCs
    python scripts/show.py bigset     # MSV000094769: files without a mapping
    python scripts/show.py genomes    # how linked genomes were matched to NCBI
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DERIVED = Path(__file__).resolve().parents[1] / "data" / "derived"


def rows(name):
    with (DERIVED / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def is_actino(row):
    return float(row["actino_fraction"] or 0) >= 0.5


def genera(top_genera):
    """'Streptomyces:2; Amycolatopsis:1' -> 'Streptomyces, Amycolatopsis'."""
    names = [re.sub(r"(\s*\([^)]*\))|(:\d+$)", "", g.strip()) for g in top_genera.split(";")]
    return ", ".join(n.replace("/", ", ") for n in names if n)


def datasets():
    projects = [p for p in rows("podp_projects.csv") if is_actino(p)]
    print(f"{'dataset':<15}{'genus':<56}{'genomes':>8}{'links':>7}{'known links':>13}")
    for p in projects:
        print(f"{p['metabolomics_id']:<15}{genera(p['top_genera'])[:54]:<56}"
              f"{p['n_actino_genomes']:>8}{p['n_genome_metabolome_links']:>7}{p['n_bgc_ms2_links']:>13}")
    n_genomes = sum(int(p["n_actino_genomes"]) for p in projects)
    single = sum(p["n_actino_genomes"] == "1" for p in projects)
    print(f"\n{len(projects)} datasets, {n_genomes} genomes; {single} datasets have only one strain")


def links():
    known = [b for b in rows("podp_bgc_ms2_links.csv") if is_actino(b)]
    by_bgc = defaultdict(list)
    for b in known:
        by_bgc[b["mibig"]].append(b)
    print(f"{len(known)} known links, but only {len(by_bgc)} different gene clusters\n")
    print(f"{'times':>5}  {'cluster':<12} example")
    for bgc, group in sorted(by_bgc.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"{len(group):>5}  {bgc:<12} {group[0]['known_link'][:45]}")
    shared = {bgc: g for bgc, g in by_bgc.items() if len({b["metabolomics_id"] for b in g}) > 1}
    short = (min((b["known_link"] for b in g), key=len) for g in shared.values())
    names = ", ".join(sorted(s.split(" and ")[0].replace(" molecular family", "").strip() for s in short))
    print(f"\nIn two different datasets: {len(shared)} clusters ({names})")


def bigset():
    files = [f for f in rows("MSV000094769_files.csv") if f["collection"] == "ccms_peak"]
    names = [f["filepath"].rsplit("/", 1)[-1] for f in files]
    sources = Counter(n.split("_")[0] for n in names)
    print(f"MSV000094769: {len(names)} spectra files, copied from {len(sources)} older datasets\n")
    for i in (0, 400, 800, 1200, 1600):
        print(f"  {names[i]}")
    print("\nNo file says which spectrum belongs to which genome.")


def genomes():
    methods = Counter(g["method"] for g in rows("genome_accessions.csv"))
    total = sum(methods.values())
    print(f"{total - methods['unresolved']} of {total} linked genomes matched to an NCBI assembly\n")
    for method, n in methods.most_common():
        print(f"{n:>5}  {method}")


COMMANDS = {"datasets": datasets, "links": links, "bigset": bigset, "genomes": genomes}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=COMMANDS)
    COMMANDS[parser.parse_args().what]()


if __name__ == "__main__":
    main()
