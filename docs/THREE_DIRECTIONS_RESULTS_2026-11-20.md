# Three research directions — implementation and validation (20 Nov 2026)

**Status: independent research proposal, not instructions attributed to Joana or Daniel.**
No change in the published DreaMS architecture, no new model training, and
no experimental proof that any new method beats NPLinker.

## Direction 1 — Validate existing spectral embeddings beyond Tahlan

**Implemented:** `scripts/representation_triage.py` summarizes recorded
A/B dataset metrics and chemical structure coverage.
`scripts/exp_f_external_validation.py` evaluates the *prespecified*
fixed 50/50 fusion on an external MassIVE dataset, excludes overlap in
first-block InChIKey between Tahlan (reference) and Chevrette (external),
and requires >=20 distinct remaining structures before estimating
descriptive correlations. All checkpoints remain frozen.

**Real run from GitHub Actions**:
https://github.com/Victoria125/priority-wur/actions/runs/44695262104

- Tahlan: 176 distinct putative chemical structures used for the initial evaluation.
- Chevrette: 21 distinct putative chemical structures before cross-dataset overlap.
- Overlapping chemical connectivity: **5**, removed from the external test.
- Unique external compounds remaining: **16**.
- Result: **INCONCLUSIVE** under the prespecified minimum of 20.
  The program correctly refuses to report externally supported
  improvement instead of presenting a numerically attractive small study.

**Limits:** different MassIVE accessions alone are not biological
independence; the models may have seen the same chemistry during
pretraining; GNPS library annotations are putative. A robust external
benchmark needs a larger independently checked chemical reference set
and preferably a different instrument/laboratory.

## Direction 2 — Combine DreaMS and MS2DeepScore

**Implemented:** `scripts/exp_e_fixed_fusion.py` uses 50/50 arithmetic
average of cosine similarity scores, no training/tuning, with molecule-level
paired exploratory uncertainty.

**Results on Tahlan, 176 structures, 15,400 dependent pairs:**

| Method | Spearman vs Morgan Tanimoto | Top-1 analog Tanimoto |
|---|---:|---:|
| DreaMS | 0.266468 | 0.508381 |
| MS2DeepScore | 0.319076 | 0.507183 |
| Fixed 50/50 score fusion | 0.340985 | 0.522952 |

Fusion – MS2DeepScore Spearman = +0.021909 with exploratory delete-one
jackknife interval [−0.007597, +0.051414]; top-1 delta = +0.015769 with
bootstrap interval [−0.004700, +0.035067]. BOTH include 0.
Therefore **no strong evidence of improvement over MS2DeepScore**.

The complete report is `docs/FUSION_RESULTS_2026-11-20.md`.

## Direction 3 — Independently specified BGC ↔ metabolomics ranking

**Implemented as a conditional prototype, NOT validated on real BGC pairs:**
`scripts/exp_g_bgc_feature_ranking.py`.

The ranking uses source-backed presence/absence observations **from the
same cultured strains** across both modalities. For every BGC×feature
hypothesis it computes:

- one-sided Fisher exact p-value for positive co-occurrence;
- signed phi coefficient to distinguish positive and negative association;
- Benjamini–Hochberg correction across all tested BGC×feature hypotheses;
- per-BGC ranking of metabolomic features, labeled **HYPOTHESIS ONLY**.

This is an intentionally transparent *non-NPLinker implementation*,
but conceptually resembles traditional co-occurrence methods and must
**never** be presented as biologically better than NPLinker.

The independent metabolomic `feature_id` could be pre-computed from GNPS,
DreaMS or MS2DeepScore networks, but no such biological association has
been inferred by this script. DreaMS embeddings are not automatically
connected to BGCs.

### Required input schemas

Complete `bgc_presence.csv`:

```csv
dataset_id,strain_id,bgc_id,present,source_ref
```

Complete `metabolite_features.csv`:

```csv
dataset_id,strain_id,feature_id,present,source_ref
```

`present` is explicitly `1` or `0` and MUST be a measured observation,
not a missing value interpreted as zero. Every strain must appear for
every BGC and for every metabolite feature; all rows have provenance.

Optional independent positive validation table:

```csv
dataset_id,bgc_id,feature_id,evidence_type,evidence_ref
```

`experimentally_verified` must be supported by documented independent
experimental evidence before interpreting any retrieval metrics. Such
evidence does NOT currently exist in curated project inputs.

CLI usage when CURATED data become available:

```bash
python dreams_molecular_families/scripts/exp_g_bgc_feature_ranking.py \
  --dataset MSV000078836 \
  --bgc-presence curated/bgc_presence.csv \
  --feature-presence curated/metabolite_features.csv \
  --verified-links curated/independently_verified_links.csv \
  --output outputs/bgc_feature_ranking.json
```

Without curated matrices, the CLI exits **2, BLOCKED, no real ranking**:

```bash
python dreams_molecular_families/scripts/exp_g_bgc_feature_ranking.py \
  --dataset MSV000078836
```

Unit tests supply **SYNTHETIC** strain matrices and deliberately known
toy associations. A perfect retrieval score on these toy examples is
a software sanity check, not biological validation.

**Important nuance:** `data/derived/podp_links.csv` contains candidate
genome-to-metabolomics-file associations from the PoDP. The files can
support manual strain-level curation; they do **not alone** define a
complete BGC presence/absence matrix or map each GNPS MS/MS scan to
the producing strain. We must verify exact sample-file links and BGC
annotations without inferring absences from missing accessions.

## Verification plan

GitHub Actions must check Python source compilation, old chemical QC
regressions, external dataset overlap filtering, fixed fusion baseline
identity, synthetic BGC-feature contingency/FDR logic, and the failure
of a real-data ranking when the curated matrices are missing.

Lean 4 proves only that **the formal safety gate** rejects missing
preconditions. It does not prove the Fisher implementation,
DreaMS accuracy, GNPS annotations, code/logic equivalence, or biology.

## Scientific next decision with Joana / Daniel

Decide if the priority is (a) an independent, chemically validated
metabolomics test set (to test whether fusion generalizes) or
(b) a small curated cross-modal strain dataset to make the ranking
biologically testable. Both are useful, but neither can be replaced
by merely adding more AI layers.
