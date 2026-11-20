"""Conservative, NPLinker-INDEPENDENT co-occurrence ranking prototype.

Purpose: rank hypotheses between BGC presence and metabolomic FEATURE presence
across the SAME curated set of bacterial strains. This is NOT a pretrained
multimodal model, not a new version of NPLinker, and is NOT validated to
outperform either NPLinker or a simple chemical-class baseline.

Every presence/absence row requires a source reference. Missing observations
must NEVER be silently interpreted as absence (0): complete cross-products
are mandatory. Biological provenance of source_ref must be manually audited.

Input schemas:
  BGC table: dataset_id,strain_id,bgc_id,present,source_ref
  Feature table: dataset_id,strain_id,feature_id,present,source_ref
  Optional positive links: dataset_id,bgc_id,feature_id,evidence_type,evidence_ref
  Only experimental positive links are eligible for descriptive retrieval metrics.

A feature may be a GNPS molecular family or a cluster derived from
DreaMS/MS2DeepScore, but it must be *defined beforehand* with a stable ID and
traceable per-strain MS/MS evidence. Comparing different feature groupings is
a future controlled experiment; no molecular families are invented here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

MIN_STRAINS = 8
POSITIVE = "experimentally_verified"
COLS = {
    "bgc": ("dataset_id", "strain_id", "bgc_id", "present", "source_ref"),
    "feature": ("dataset_id", "strain_id", "feature_id", "present", "source_ref"),
    "positives": ("dataset_id", "bgc_id", "feature_id", "evidence_type", "evidence_ref"),
}


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path}: missing required columns: {sorted(set(required)-set(reader.fieldnames or []))}")
        rows = []
        for line, row in enumerate(reader, 2):
            clean = {k: (v or "").strip() for k, v in row.items() if k}
            if any(not clean.get(column) for column in required):
                raise ValueError(f"{path}:{line}: blank required value; missing is NOT absence")
            rows.append(clean)
        return rows


def presence_matrix(records, dataset_id: str, entity: str):
    """Reject incomplete, duplicated or inconsistent source-backed data."""
    field = "bgc_id" if entity == "bgc" else "feature_id"
    rows = [r for r in records if r.get("dataset_id") == dataset_id]
    if not rows:
        raise ValueError(f"No curated {entity} observations for dataset {dataset_id}")
    cells = {}
    strains, ids = set(), set()
    for row in rows:
        for col in COLS[entity]:
            if not row.get(col) or not str(row[col]).strip():
                raise ValueError("Required evidence field absent: " + col)
        if row["present"] not in ("0", "1"):
            raise ValueError("present must be explicit 0 or 1 (unknown is NOT 0)")
        key = (row["strain_id"], row[field])
        if key in cells:
            raise ValueError(f"Duplicate observation {entity}: {key}")
        cells[key] = int(row["present"])
        strains.add(key[0])
        ids.add(key[1])
    if len(strains) < MIN_STRAINS:
        raise ValueError(f"Only {len(strains)} strains; minimum {MIN_STRAINS} for exploratory ranking")
    for s in strains:
        for eid in ids:
            if (s, eid) not in cells:
                raise ValueError(f"Incomplete {entity} matrix: missing {s}/{eid}; never impute absence")
    return sorted(strains), sorted(ids), cells


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """BH-FDR; all pairwise BGC×feature hypotheses belong to one test family."""
    n = len(pvalues)
    if not n:
        return []
    if any(not math.isfinite(float(p)) or p < 0 or p > 1 for p in pvalues):
        raise ValueError("Invalid P values")
    order = sorted(range(n), key=lambda i: pvalues[i])
    q = [1.0] * n
    last = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        last = min(last, pvalues[i] * n / rank)
        q[i] = last
    return q


def phi_coefficient(n11: int, n10: int, n01: int, n00: int) -> float:
    """Signed association; positive means gene and feature co-occur."""
    d = (n11+n10)*(n01+n00)*(n11+n01)*(n10+n00)
    if not d:
        return 0.0
    return (n11*n00 - n10*n01) / math.sqrt(d)


def ranking(bgc_records, feature_records, dataset_id: str):
    bgc_strains, bgcs, bgc_cells = presence_matrix(bgc_records, dataset_id, "bgc")
    feature_strains, features, feature_cells = presence_matrix(feature_records, dataset_id, "feature")
    if bgc_strains != feature_strains:
        raise ValueError("BGC and metabolomics modalities must cover EXACTLY the same curated strains")
    strains = bgc_strains
    rows = []
    for b in bgcs:
        bv = [bgc_cells[(strain, b)] for strain in strains]
        if len(set(bv)) != 2:
            raise ValueError(f"Uninformative BGC presence: {b} (no 0/1 variation)")
        for feature in features:
            fv = [feature_cells[(strain, feature)] for strain in strains]
            if len(set(fv)) != 2:
                raise ValueError(f"Uninformative metabolomic feature: {feature} (no 0/1 variation)")
            n11 = sum(x == 1 and y == 1 for x, y in zip(bv, fv))
            n10 = sum(x == 1 and y == 0 for x, y in zip(bv, fv))
            n01 = sum(x == 0 and y == 1 for x, y in zip(bv, fv))
            n00 = sum(x == 0 and y == 0 for x, y in zip(bv, fv))
            effect = phi_coefficient(n11, n10, n01, n00)
            p = float(fisher_exact([[n11, n10], [n01, n00]], alternative="greater").pvalue)
            rows.append({
                "bgc_id": b, "feature_id": feature,
                "n11": n11, "n10": n10, "n01": n01, "n00": n00,
                "phi": effect, "fisher_one_sided_p": p,
                "label": "HYPOTHESIS_ONLY_NOT_BIOLOGICALLY_CONFIRMED",
            })
    q = benjamini_hochberg([x["fisher_one_sided_p"] for x in rows])
    for item, qi in zip(rows, q):
        item["bh_fdr_q"] = qi
    by_bgc = defaultdict(list)
    for item in rows:
        by_bgc[item["bgc_id"]].append(item)
    result = []
    for b, pairs in sorted(by_bgc.items()):
        # Positive phi first; p-value and effect size then break ties.
        sorted_pairs = sorted(pairs, key=lambda row: (
            row["phi"] <= 0, row["fisher_one_sided_p"], -row["phi"], row["feature_id"]
        ))
        for rank, item in enumerate(sorted_pairs, 1):
            item["rank_within_bgc"] = rank
            result.append(item)
    return {
        "dataset_id": dataset_id, "curated_strains": len(strains),
        "bgc_count": len(bgcs), "metabolomic_feature_count": len(features),
        "ranked_candidates": result,
        "ground_truth_status": "none; ranking is NOT validated",
        "method": "one-sided Fisher co-occurrence + phi effect, BH correction across all tested pairs",
        "caveats": [
            "Absence (0) must be supported by a real experiment; it is not inferred from missing rows.",
            "A statistically associated feature may be correlated with other BGCs or bacterial lineage.",
            "Metabolomic features may be different chemical ions of the same compound.",
            "Source references and strain identities require independent manual audit.",
            "No evidence this baseline is better than NPLinker or competing methods.",
        ],
    }


def evaluate_known_positives(rank_report: dict, labels: list[dict], min_labels=3) -> dict:
    """Descriptive positive-only recall and reciprocal rank (NOT verified negatives)."""
    dataset = rank_report["dataset_id"]
    confirmed = [
        r for r in labels
        if r.get("dataset_id") == dataset and r.get("evidence_type") == POSITIVE
    ]
    seen = set()
    for r in confirmed:
        if not all(r.get(k) for k in COLS["positives"]):
            raise ValueError("Verified positives must have dataset, pair, type, source")
        pair = (r["bgc_id"], r["feature_id"])
        if pair in seen:
            raise ValueError("Duplicate experimental positive: " + str(pair))
        seen.add(pair)
    if len(confirmed) < min_labels:
        return {"status": "NOT EVALUABLE: too few independently asserted positive links",
                "verified_positives": len(confirmed), "metrics": None}
    ranks = {(r["bgc_id"], r["feature_id"]): r["rank_within_bgc"]
             for r in rank_report["ranked_candidates"]}
    missing = sorted(pair for pair in seen if pair not in ranks)
    if missing:
        raise ValueError(f"Experimental positives outside ranked candidate universe: {missing}")
    rr = [ranks[pair] for pair in sorted(seen)]
    return {
        "status": "DESCRIPTIVE ONLY; curator must independently verify source references",
        "verified_positives": len(rr),
        "metrics": {
            "recall_at_1": sum(r <= 1 for r in rr) / len(rr),
            "recall_at_5": sum(r <= 5 for r in rr) / len(rr),
            "mean_reciprocal_rank": sum(1 / r for r in rr) / len(rr),
        },
        "caveat": "No validated negative links, biological proof, independent split or supervised model.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bgc-presence", type=Path)
    parser.add_argument("--feature-presence", type=Path)
    parser.add_argument("--verified-links", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bgc_presence is None or args.feature_presence is None:
        report = {
            "status": "BLOCKED: curated presence/absence tables were not supplied",
            "dataset_id": args.dataset,
            "missing": [
                "Source-backed complete strain × BGC presence/absence",
                "Source-backed complete strain × metabolomic-feature presence/absence",
            ],
            "no_ranking_generated": True,
        }
        code = 2
    else:
        bgcs = read_csv(args.bgc_presence, COLS["bgc"])
        features = read_csv(args.feature_presence, COLS["feature"])
        report = ranking(bgcs, features, args.dataset)
        report["positive_link_evaluation"] = (
            evaluate_known_positives(report, read_csv(args.verified_links, COLS["positives"]))
            if args.verified_links is not None else
            {"status": "NOT EVALUATED: no independent biological positives supplied"}
        )
        code = 0
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
