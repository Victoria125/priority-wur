"""Inventory of the Paired Omics Data Platform (PoDP).

Downloads every PoDP project record, estimates how actinobacterial each one
is from its genome labels, and writes three tables:

    data/derived/podp_projects.csv        one row per project
    data/derived/podp_links.csv           one row per genome <-> metabolomics-file link
    data/derived/podp_bgc_ms2_links.csv   one row per known BGC <-> MS2 link

Raw JSON is cached in data/raw/podp/ so reruns don't hit the API again.
Standard library only.

Usage:
    python scripts/podp_inventory.py            # use cache when present
    python scripts/podp_inventory.py --refresh  # re-download everything
"""
import argparse
import csv
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://pairedomicsdata.bioinformatics.nl/api/projects"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "podp"
OUT = ROOT / "data" / "derived"

# Genera of the phylum Actinomycetota. Natural-product producers first, then
# common host-associated genera so mixed collections get a fair fraction.
ACTINO_GENERA = {
    "actinokineospora", "actinomadura", "actinoplanes", "allokutzneria",
    "amycolatopsis", "catenulispora", "dactylosporangium", "frankia",
    "kitasatospora", "kribbella", "kutzneria", "lentzea", "microbispora",
    "micromonospora", "nocardia", "nocardioides", "nocardiopsis", "nonomuraea",
    "planomonospora", "pseudonocardia", "rhodococcus", "saccharomonospora",
    "saccharopolyspora", "salinispora", "streptacidiphilus", "streptomyces",
    "streptosporangium", "verrucosispora", "gordonia", "mycobacterium",
    "corynebacterium", "actinomyces", "arthrobacter", "bifidobacterium",
    "brevibacterium", "collinsella", "cutibacterium", "eggerthella", "kocuria",
    "microbacterium", "micrococcus", "propionibacterium", "rothia",
}

# Projects whose genome labels are bare strain IDs (no genus), curated by hand
# from the species list of the MassIVE record. Keyed by metabolomics ID.
MANUAL_TAXA = {
    # Chevrette et al. 2019, insect-associated actinobacteria
    "MSV000084723": "Streptomyces/Pseudonocardia/Gordonia/Kitasatospora (MassIVE)",
}


def get_json(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "priority-wur-inventory/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def cached(path, url, refresh):
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = get_json(url)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def genus_of(label):
    """First token of a genome label that is a known genus, else ''."""
    for token in re.findall(r"[A-Za-z]+", label or ""):
        if token.lower() in ACTINO_GENERA:
            return token.capitalize()
    first = re.match(r"\s*([A-Z][a-z]{2,})\b", label or "")
    return first.group(1) if first else ""


def accession_of(genome):
    gid = genome.get("genome_ID") or {}
    for key in ("RefSeq_accession", "GenBank_accession", "JGI_Genome_ID"):
        if gid.get(key):
            return key.split("_")[0], str(gid[key]).strip()
    return "", ""


def summarise(pid, listing, record):
    personal = record.get("personal") or {}
    metab = record.get("metabolomics") or {}
    exp = record.get("experimental") or {}
    genomes = record.get("genomes") or []
    links = record.get("genome_metabolome_links") or []
    bgc_links = record.get("BGC_MS2_links") or []

    genera = Counter(genus_of(g.get("genome_label", "")) for g in genomes)
    n_actino = sum(n for g, n in genera.items() if g.lower() in ACTINO_GENERA)
    manual = MANUAL_TAXA.get(listing.get("metabolite_id", ""))
    if manual and set(genera) == {""}:
        genera, n_actino = Counter({manual: len(genomes)}), len(genomes)
    instruments = sorted({
        str((m.get("instrumentation") or {}).get("instrument", "")).strip()
        for m in exp.get("instrumentation_methods") or []
    } - {""})
    id_types = Counter(accession_of(g)[0] or "none" for g in genomes)

    return {
        "podp_id": pid,
        "metabolomics_id": listing.get("metabolite_id", ""),
        "pi": listing.get("PI_name", ""),
        "pi_institution": personal.get("PI_institution", ""),
        "n_genomes": len(genomes),
        "n_actino_genomes": n_actino,
        "actino_fraction": round(n_actino / len(genomes), 2) if genomes else 0,
        "top_genera": "; ".join(f"{g or '?'}:{n}" for g, n in genera.most_common(4)),
        "n_genome_metabolome_links": len(links),
        "n_linked_genomes": len({l.get("genome_label") for l in links}),
        "n_bgc_ms2_links": len(bgc_links),
        "n_growth_conditions": len(exp.get("sample_preparation") or []),
        "n_extraction_methods": len(exp.get("extraction_methods") or []),
        "instruments": "; ".join(instruments),
        "genome_id_types": "; ".join(f"{k}:{v}" for k, v in id_types.items()),
        "publications": str(metab.get("publications", "")).strip(),
    }


def link_rows(pid, listing, record):
    genomes = {g.get("genome_label", "").strip(): g for g in record.get("genomes") or []}
    for link in record.get("genome_metabolome_links") or []:
        label = (link.get("genome_label") or "").strip()
        id_type, acc = accession_of(genomes.get(label, {}))
        yield {
            "podp_id": pid,
            "metabolomics_id": listing.get("metabolite_id", ""),
            "genome_label": label,
            "genus": genus_of(label),
            "genome_id_type": id_type,
            "genome_accession": acc,
            "biosample": str(genomes.get(label, {}).get("BioSample_accession", "")).strip(),
            "metabolomics_file": link.get("metabolomics_file", ""),
            "growth_condition": link.get("sample_preparation_label", ""),
            "extraction": link.get("extraction_method_label", ""),
            "instrument_method": link.get("instrumentation_method_label", ""),
        }


def bgc_ms2_rows(pid, listing, record, actino_fraction):
    for link in record.get("BGC_MS2_links") or []:
        bgc = link.get("BGC_ID") or {}
        yield {
            "podp_id": pid,
            "metabolomics_id": listing.get("metabolite_id", ""),
            "pi": listing.get("PI_name", "").strip(),
            "actino_fraction": actino_fraction,
            "mibig": (bgc.get("MIBiG_number") or bgc.get("similar_MIBiG_number") or "").strip(),
            "bgc_type": bgc.get("BGC", ""),
            "known_link": " ".join((link.get("known_link") or "").split()),
            "link_type": link.get("link", ""),
            "has_smiles": bool((link.get("SMILES") or "").strip()),
            "omics_evidence": (link.get("omics_based_evidence") or {}).get("omics_based_evidence_type", ""),
        }


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    listing = cached(RAW / "_projects.json", API, args.refresh)
    projects, links, bgc_links = [], [], []
    for entry in listing["data"]:
        pid = entry["_id"]
        record = cached(RAW / f"{pid}.json", f"{API}/{pid}", args.refresh)
        record = record.get("project", record)  # tolerate a wrapped payload
        projects.append(summarise(pid, entry, record))
        links.extend(link_rows(pid, entry, record))
        bgc_links.extend(bgc_ms2_rows(pid, entry, record, projects[-1]["actino_fraction"]))

    projects.sort(key=lambda p: (-p["actino_fraction"], -p["n_genome_metabolome_links"]))
    write_csv(OUT / "podp_projects.csv", projects)
    write_csv(OUT / "podp_links.csv", links)
    write_csv(OUT / "podp_bgc_ms2_links.csv", bgc_links)

    actino = [p for p in projects if p["actino_fraction"] >= 0.5]
    print(f"PoDP projects: {len(projects)} (listing says {listing.get('total')})")
    print(f"Actinobacteria-dominant (>=50% of genomes): {len(actino)} projects, "
          f"{sum(p['n_actino_genomes'] for p in actino)} genomes, "
          f"{sum(p['n_genome_metabolome_links'] for p in actino)} genome-metabolome links")
    print(f"\n{'metabolomics_id':<16}{'genomes':>8}{'links':>7}{'bgc-ms2':>8}  top genera / PI")
    for p in actino[:20]:
        print(f"{p['metabolomics_id']:<16}{p['n_genomes']:>8}{p['n_genome_metabolome_links']:>7}"
              f"{p['n_bgc_ms2_links']:>8}  {p['top_genera']} / {p['pi']}")
    actino_bgc = [b for b in bgc_links if b["actino_fraction"] >= 0.5]
    print(f"\nKnown BGC-MS2 links in those projects: {len(actino_bgc)}, "
          f"covering {len({b['mibig'] for b in actino_bgc})} distinct MIBiG BGCs")
    print(f"\nWrote {OUT / 'podp_projects.csv'}, {OUT / 'podp_links.csv'} "
          f"and {OUT / 'podp_bgc_ms2_links.csv'}")


if __name__ == "__main__":
    main()
