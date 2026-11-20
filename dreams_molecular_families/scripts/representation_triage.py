"""Evidence-aware triage of existing DreaMS/MS2DeepScore results.

This reports ALREADY-COMPUTED results; it never trains/inferences on spectra.
Only known input sections from scripts/report.py are accepted, to avoid
accidentally treating a sensitivity table as the main protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "dreams_molecular_families" / "RESULTS.md"
PROJECTS = ROOT / "data" / "derived" / "podp_projects.csv"

MIN_PRIMARY_STRUCTURES = 100
MIN_EXTERNAL_STRUCTURES = 20
METHODS = ("DreaMS", "MS2DeepScore", "binned cosine")


def section(text: str, heading: str) -> str:
    marker = "## " + heading
    start = text.find(marker + "\n")
    if start < 0:
        raise ValueError("Missing report section: " + heading)
    rest = text[start + len(marker) + 1:]
    next_heading = re.search(r"^## ", rest, flags=re.MULTILINE)
    return rest[:next_heading.start()] if next_heading else rest


def table(text: str) -> tuple[list[str], dict[str, list[str]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if not lines:
        raise ValueError("Missing markdown table")
    def cells(line):
        return [x.strip() for x in line.strip().strip("|").split("|")]
    header = cells(lines[0])
    if len(header) < 2 or header[0] != "" or not all(header[1:]):
        raise ValueError("Unexpected report table header")
    if len(set(header[1:])) != len(header[1:]):
        raise ValueError("Duplicate run name")
    data = {}
    for line in lines[1:]:
        row = cells(line)
        if row[0].startswith("---"):
            continue
        if len(row) != len(header):
            raise ValueError("Row does not match report header: " + str(row[:2]))
        if row[0] in data:
            raise ValueError("Duplicate report metric: " + row[0])
        data[row[0]] = row[1:]
    return header[1:], data


def read_metric(names, rows, name: str, run: str, *, allow_missing=False):
    if name not in rows:
        if allow_missing:
            return None
        raise ValueError("Metric absent: " + name)
    cell = rows[name][names.index(run)]
    if not cell:
        if allow_missing:
            return None
        raise ValueError(f"Missing required {name} for {run}")
    return cell


def positive_int(cell: str, description: str) -> int:
    try:
        n = int(cell)
    except (ValueError, TypeError):
        raise ValueError("Expected integer for " + description) from None
    if n < 0:
        raise ValueError("Negative count for " + description)
    return n


def float_metric(cell: str | None) -> float | None:
    if cell is None:
        return None
    n = float(cell)
    if not (-1 <= n <= 1):
        raise ValueError("Out-of-range correlation/recovery: " + str(n))
    return n


def ci_includes_zero(cell: str | None) -> bool | None:
    if not cell:
        return None
    match = re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s*\[([+-]?\d+(?:\.\d+)?),\s*([+-]?\d+(?:\.\d+)?)\]", cell)
    if not match:
        raise ValueError("Unrecognized confidence interval: " + cell)
    low, high = map(float, match.groups())
    if low > high:
        raise ValueError("Confidence interval bounds reversed")
    return low <= 0 <= high


def project_index(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    out = {}
    for row in records:
        accession = row["metabolomics_id"]
        n_genomes = positive_int(row["n_genomes"], "PoDP genomes")
        n_links = positive_int(row["n_bgc_ms2_links"], "PoDP linked records")
        v = out.setdefault(accession, {"n_genomes": 0, "podp_reported_bgc_ms2": 0, "project_entries": 0})
        v["n_genomes"] = max(v["n_genomes"], n_genomes)
        # This is an INVENTORY sum, not independent spectrum-level truth.
        v["podp_reported_bgc_ms2"] += n_links
        v["project_entries"] += 1
    return out


def evaluate(report_text: str, projects: dict) -> dict:
    names, dataset = table(section(report_text, "Datasets"))
    main_names, main = table(section(report_text, "Main numbers (DreaMS protocol)"))
    if names != main_names:
        raise ValueError("Dataset and main-result columns are not aligned")
    runs = []
    for run in names:
        accession = read_metric(names, dataset, "MassIVE dataset (PoDP)", run)
        n = positive_int(read_metric(names, main, "B independent compounds after all checks", run, allow_missing=True) or "0", "independent structures")
        accepted = positive_int(read_metric(names, dataset, "spectra (protocol)", run), "protocol spectra")
        raw = positive_int(read_metric(names, dataset, "spectra with peaks", run), "raw spectra")
        if accepted > raw:
            raise ValueError("More accepted spectra than originals: " + run)
        if accepted == 0 and n:
            raise ValueError("Structures reported with zero accepted spectra: " + run)
        domain_coverage = accepted / raw if raw else 0
        provenance = read_metric(names, dataset, "spectra sources (provenance check)", run)
        scores = {}
        for method in METHODS:
            scores[method] = float_metric(read_metric(names, main, f"A1 family recovery@5: {method}", run, allow_missing=True))
        structural_scores = {
            method: float_metric(read_metric(names, main, f"B1 Spearman with Tanimoto: {method}", run, allow_missing=True))
            for method in ("DreaMS", "MS2DeepScore", "modified cosine")
        }
        if n >= MIN_PRIMARY_STRUCTURES and any(v is None for v in structural_scores.values()):
            raise ValueError("Primary chemical candidate lacks comparison scores: " + run)
        chemical_role = (
            "primary structural benchmark candidate" if n >= MIN_PRIMARY_STRUCTURES and accepted
            else "exploratory external structural set" if n >= MIN_EXTERNAL_STRUCTURES and accepted
            else "insufficient independent structures for primary conclusions"
        )
        # Do not use GNPS agreement as independent molecular-structure ground truth.
        a1_ci = ci_includes_zero(read_metric(names, main, "A1 DreaMS minus binned cosine (95% CI)", run, allow_missing=True))
        b_ci = ci_includes_zero(read_metric(names, main, "B1 DreaMS minus MS2DeepScore (95% CI)", run, allow_missing=True))
        summary = {
            "run": run, "massive_accession": accession,
            "protocol_spectra": accepted, "spectra_with_peaks": raw,
            "in_domain_fraction": round(domain_coverage, 4),
            "distinct_chemical_structures": n,
            "chemical_benchmark_role": chemical_role,
            "a1_family_recovery_at_5": scores,
            "a1_dreams_vs_binned_ci_includes_zero": a1_ci,
            "b1_spearman_with_tanimoto": structural_scores,
            "b1_dreams_vs_ms2deepscore_ci_includes_zero": b_ci,
            "spectral_provenance": provenance,
            "podp_inventory": projects.get(accession, {}),
            "strain_resolved_linking_validation": "NOT ESTABLISHED: curated spectrum–strain–BGC mapping required",
        }
        runs.append(summary)

    candidates = [r["run"] for r in runs if r["chemical_benchmark_role"].startswith("primary")]
    if not candidates:
        raise ValueError("No dataset meets prespecified primary chemical reference size")
    return {
        "source": "Existing RESULTS.md main protocol; no new embedding inference",
        "prespecified_thresholds": {
            "primary_distinct_structures": MIN_PRIMARY_STRUCTURES,
            "exploratory_distinct_structures": MIN_EXTERNAL_STRUCTURES,
        },
        "primary_structural_benchmark_candidates": candidates,
        "ready_for_bgc_spectrum_linking": [],
        "why_not_ready": "PoDP inventory and GNPS metadata do not supply independently curated spectrum-to-strain-to-BGC joins",
        "limits": [
            "Library structures remain putative after chemical QC, not experimental identification for each spectrum.",
            "GNPS family recovery is not independent chemical ground truth because GNPS network construction uses cosine.",
            "Bootstrap intervals for A/B do not replace tests on independent held-out laboratories/families.",
            "This report summarizes recorded results; it does not rerun DreaMS or MS2DeepScore.",
        ],
        "runs": runs,
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# Dataset-first triage of spectral embeddings",
        "",
        "**Derived from existing 18 November 2026 RESULTS.md; no new model run.**",
        "Candidates are for research, not evidence of verified biology.",
        "",
        "| Run | Protocol spectra | In-domain | Distinct structures | Suggested role |",
        "|---|---:|---:|---:|---|",
    ]
    for r in result["runs"]:
        lines.append(f'| {r["run"]} | {r["protocol_spectra"]} | {r["in_domain_fraction"]:.0%} | {r["distinct_chemical_structures"]} | {r["chemical_benchmark_role"]} |')
    lines += ["", "## Conclusions supported by the recorded results", ""]
    lines += [
        "- Primary structure-focused analysis: " + ", ".join(result["primary_structural_benchmark_candidates"]) + ".",
        "- None of the six public datasets is independently verified as a strain-resolved BGC–MS/MS benchmark.",
        "- A1 is GNPS family agreement, NOT independent validation of chemical structures.",
        "- Compare DreaMS, MS2DeepScore and cosine per dataset; do not infer improvement from a numerically higher score when the paired interval includes zero.",
        "- Include the in-domain fraction, false annotation risk and exact MS/MS provenance in every conclusion.",
        "",
        "## Experiments that require additional data",
        "",
        "1. Hold out whole chemical families or laboratories (not random spectrum pairs) when sufficient ground truth exists.",
        "2. Add pre-specified DreaMS/MS2DeepScore fusion only as a hypothesis, reporting paired confidence intervals versus both components and cosine.",
        "3. Only consider independent BGC–MS/MS ranking after explicit spectrum→strain and strain→BGC evidence, with known positives held out.",
        "4. Do not prioritize federated learning before establishing a validated prediction task.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--projects", type=Path, default=PROJECTS)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = evaluate(args.results.read_text(encoding="utf-8"), project_index(args.projects))
    markdown = render_markdown(report)
    print(markdown)
    for path, body in ((args.markdown_out, markdown + "\n"), (args.json_out, json.dumps(report, indent=2) + "\n")):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
