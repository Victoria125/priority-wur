"""Conservative evidence audit for a *possible* BGC–MS/MS linking benchmark.

The PoDP BGC–MS2 inventory documents literature associations. It does NOT
provide an automatically verified spectrum ↔ strain ↔ BGC mapping.

This module never infers strain IDs from file names, chemical names, or MIBiG
similarity. Required cross-modal identifiers must come from curated CSV files.

An all-green result only means *internally consistent curated assertions*;
it is NOT proof that the asserted biology is true, or that a model works.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PODP = ROOT / "data" / "derived" / "podp_bgc_ms2_links.csv"

SCHEMAS = {
    "strain_spectra": ("dataset_id", "spectrum_id", "strain_id", "source_ref"),
    "strain_bgcs": ("dataset_id", "strain_id", "bgc_id", "source_ref"),
    "verified_links": ("dataset_id", "spectrum_id", "bgc_id", "evidence_type", "evidence_ref"),
}
POSITIVE_EVIDENCE = "experimentally_verified"


def load_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    """Reject missing headers or blank required fields. Never silently fill IDs."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = []
        for n, original in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in original.items() if k}
            blanks = [k for k in required if not row.get(k)]
            if blanks:
                raise ValueError(f"{path}:{n}: blank required fields: {', '.join(blanks)}")
            rows.append(row)
        return rows


def audit(podp_links: list[dict[str, str]], dataset_id: str,
          strain_spectra: list[dict[str, str]] | None = None,
          strain_bgcs: list[dict[str, str]] | None = None,
          verified_links: list[dict[str, str]] | None = None) -> dict:
    """Audit provenance and joins, WITHOUT assigning biological truth to PoDP rows.

    None means a curated evidence table was NOT provided. Empty lists mean a
    supplied table had no rows. Either fails the evidence completeness gate.
    """
    source = [r for r in podp_links if r.get("metabolomics_id") == dataset_id]
    issues: list[str] = []
    mapped_spectra: dict[str, str] = {}
    bgc_by_strain: dict[str, set[str]] = defaultdict(set)
    truth_pairs: set[tuple[str, str]] = set()

    if not source:
        issues.append("No PoDP BGC–MS2 inventory rows for the selected dataset")
    if strain_spectra is None:
        issues.append("Missing curated spectrum-to-strain mapping")
    if strain_bgcs is None:
        issues.append("Missing curated strain-to-BGC mapping")
    if verified_links is None:
        issues.append("Missing independently sourced BGC-to-spectrum evidence table")

    if strain_spectra is not None:
        for r in strain_spectra:
            if r["dataset_id"] != dataset_id:
                continue
            spectrum, strain = r["spectrum_id"], r["strain_id"]
            if spectrum in mapped_spectra and mapped_spectra[spectrum] != strain:
                issues.append(f"Conflicting strain assignment for spectrum {spectrum}")
            mapped_spectra[spectrum] = strain
        if not mapped_spectra:
            issues.append("No spectrum-to-strain records for the dataset")

    if strain_bgcs is not None:
        for r in strain_bgcs:
            if r["dataset_id"] == dataset_id:
                bgc_by_strain[r["strain_id"]].add(r["bgc_id"])
        if not bgc_by_strain:
            issues.append("No strain-to-BGC records for the dataset")

    evidence_types = Counter()
    if verified_links is not None:
        for r in verified_links:
            if r["dataset_id"] != dataset_id:
                continue
            evidence_types[r["evidence_type"]] += 1
            if r["evidence_type"] != POSITIVE_EVIDENCE:
                continue
            spectrum, bgc = r["spectrum_id"], r["bgc_id"]
            if spectrum not in mapped_spectra:
                issues.append(f"Verified link lacks a spectrum-to-strain mapping: {spectrum}")
                continue
            strain = mapped_spectra[spectrum]
            if bgc not in bgc_by_strain.get(strain, set()):
                issues.append(f"Verified link lacks corresponding BGC in its assigned strain: {spectrum}/{bgc}")
                continue
            truth_pairs.add((spectrum, bgc))
        if not evidence_types:
            issues.append("No BGC-to-spectrum evidence rows for this dataset")
        if not truth_pairs:
            issues.append("No internally consistent experimentally verified links")

    strains = set(mapped_spectra.values()) & set(bgc_by_strain)
    profiles = {tuple(sorted(bgc_by_strain[strain])) for strain in strains}
    if len(strains) < 2:
        issues.append("Fewer than two strains with both spectra and BGC data")
    if len(profiles) < 2:
        issues.append("No observable BGC-presence variation across mapped strains")

    return {
        "dataset_id": dataset_id,
        "podp_reported_rows": len(source),
        "podp_distinct_mibig_ids": len({r.get("mibig") for r in source if r.get("mibig")}),
        "mapped_spectra": len(mapped_spectra),
        "mapped_strains_with_both_modalities": len(strains),
        "distinct_bgc_profiles": len(profiles),
        "independently_curated_evidence_types": dict(sorted(evidence_types.items())),
        "internally_consistent_verified_pairs": len(truth_pairs),
        "ready_for_manual_benchmark_review": not issues,
        "issues": list(dict.fromkeys(issues)),
        "caveat": "Only checks curated IDs, provenance fields, and joins. Not biological validation, model evaluation, leakage prevention, or clinical evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="MassIVE accession e.g. MSV000078836")
    parser.add_argument("--podp-links", type=Path, default=DEFAULT_PODP)
    parser.add_argument("--strain-spectra", type=Path, help="Curated dataset_id,spectrum_id,strain_id,source_ref CSV")
    parser.add_argument("--strain-bgcs", type=Path, help="Curated dataset_id,strain_id,bgc_id,source_ref CSV")
    parser.add_argument("--verified-links", type=Path, help="Curated dataset_id,spectrum_id,bgc_id,evidence_type,evidence_ref CSV")
    parser.add_argument("--output", type=Path, help="Optional JSON audit output")
    parser.add_argument("--require-ready", action="store_true", help="Exit 2 unless evidence-completeness checks pass")
    args = parser.parse_args()

    links = load_rows(args.podp_links, ("metabolomics_id", "mibig", "bgc_type", "link_type"))
    def optional_rows(path, name):
        return None if path is None else load_rows(path, SCHEMAS[name])
    result = audit(
        links, args.dataset,
        optional_rows(args.strain_spectra, "strain_spectra"),
        optional_rows(args.strain_bgcs, "strain_bgcs"),
        optional_rows(args.verified_links, "verified_links"),
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(payload, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 2 if args.require_ready and not result["ready_for_manual_benchmark_review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
