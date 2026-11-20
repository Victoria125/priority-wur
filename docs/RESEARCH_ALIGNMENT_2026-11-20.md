# Research alignment and evidence-gated next steps (20 November 2026)

## Purpose and provenance of decisions

**Scientific direction (proposed, not agreed with Joana or Daniel):**
First identify a traceable actinobacterial dataset and establish a reliable
metabolomics representation; only then attempt BGC–MS/MS link ranking.
Distributed/federated training is **deferred**, not the immediate milestone.

Sources, rather than attributing unrecorded preferences to a researcher:
- Joana Tirana's 6 October 2026 email: focus first on a *good dataset for
  training models* and *embedding/foundation models for each omic layer*,
  emphasizing actinobacteria. She shared NPLinker, COMMBAT, multi-omics and
  federated-learning references; these were options, not approval of a model.
- The DreaMS article Joana shared on 19 October:
  https://www.nature.com/articles/s41587-025-02663-3
- Public methodological examples, **not a claim these are her choices for
  PRIORITY**: https://github.com/jtirana98/Hydra-CF-in-SFL
  and https://github.com/jtirana98/SFL-workflow-optimization .
  They document client partitions, reference methods, protocols, logging
  and the challenges of cross-client heterogeneity.
- Initial dataset/representation inventory: `docs/01_initial_exploration.md`.
- Current analyses: `dreams_molecular_families/RESULTS.md`, `README.md`
  and `HANDOFF_FOR_JOANA_2026-11-10.md`.

## One scientific question, not four unrelated demos

**Main question:** Which available MS/MS representation is most dependable on
actinobacterial datasets, and do we have enough independent sample-to-strain
evidence to test whether it helps rank BGC–metabolite links?

The first two experiments (A/B) are **spectral representation benchmarks**.
Experiments C/D are **exploratory diagnostics**, not a demonstrated linking
improvement. NPLinker is a *future baseline*, not a completed integration.

### Stage 1: data evidence and eligibility (now)

Before ranking a gene cluster against a spectrum, establish these mappings
from independent curated records, keeping a source reference for every row:

1. `dataset_id,spectrum_id,strain_id,source_ref` — sample/spectrum provenance
   joined explicitly to the corresponding strain.
2. `dataset_id,strain_id,bgc_id,source_ref` — genome/BGC presence for each strain.
3. `dataset_id,spectrum_id,bgc_id,evidence_type,evidence_ref` —
   held-out positive evidence. Use `experimentally_verified` **only when the
   curator has documentary experimental support**, otherwise `reported`.

Run the fail-closed evidence check, using Crüsemann merely as a **candidate**:

```bash
python dreams_molecular_families/scripts/linking_evidence.py --dataset MSV000078836
```

This MUST report not ready with only PoDP data. PoDP known-link records,
name matches in GNPS libraries, MIBiG 'similar BGC' identifiers, and a
MassIVE accession are not sufficient to join an individual MS/MS spectrum
to the producing strain and the corresponding genomic BGC.

When curated tables are prepared, run:

```bash
python dreams_molecular_families/scripts/linking_evidence.py \
  --dataset MSV000078836 \
  --strain-spectra curated/strain_spectra.csv \
  --strain-bgcs curated/strain_bgcs.csv \
  --verified-links curated/verified_links.csv \
  --output linking_readiness.json --require-ready
```

The gate checks internal consistency and presence of sourced assertions.
It **cannot validate the underlying biology** or certify predictive accuracy.
Do not use filename heuristics or the GNPS library compound name as a
surrogate for verified strain/sample mapping. Review uncertainty manually.

### Stage 2: make experiments A/B answer a coherent question

- Preserve input and model provenance; compare DreaMS, MS2DeepScore,
  matched cosine methods and a mass-only/random control on equal subsets.
- Distinguish GNPS-family *agreement* from independent chemical truth.
  GNPS networks arise from spectral cosine, creating a circularity caveat.
- Use chemically checked structures for B. Tahlan currently has 176
  independent compounds; smaller sets must not support strong claims.
- Report uncertainty and per-dataset sample counts. Do not interpret
  Spearman as accuracy or non-significant differences as improvements.
- Diagnose domain exclusions (charge, m/z limits, peak quality) before
  claiming model failure or model superiority.
- Consider future leave-lab-out/held-out molecular-family evaluations,
  but ensure meaningful independent labels first. DreaMS pretraining
  might overlap publicly annotated chemicals, so check potential leakage.

### Stage 3: linking research only after Stage 1

- Define specific strain-matched positive BGC–MS/MS examples.
- Run NPLinker (Metcalf/Rosetta/NPClassScore as appropriate) as a baseline
  *after* the strain mapping is validated.
- Then evaluate whether a DreaMS- or MS2DeepScore-based family score,
  used alongside other biological evidence, improves ranking.
- Evaluate recall@k / MRR on independent positive links and document
  uncertain negatives, split strategy, possible family-level leakage.
- If known links and strain mappings are insufficient, report that
  limitation; **do not manufacture ground truth or performance**.

### Stage 4: federated learning — DEFERRED

Current Experiment C is a small 1024-to-256 linear adapter on frozen
DreaMS embeddings with heterogeneous partitions. It is not a new DreaMS
architecture and does not implement SplitFed. Its training budgets are not
fully compute-matched. Do **not** claim that FL outperforms centralised
training, nor extend FL just to add complexity.

Revisit FL only when a useful supervised objective, independent labels,
clear client/data ownership and a centralised control exist. At that
stage compare matched compute, client participation, non-IID partitions,
privacy requirements and generalization. These requirements reflect
documented good research practice, not a statement of Joana's approval.

## Summary for next meeting

1. **Completed:** DreaMS tutorial reproduction, six-dataset MS/MS analysis,
   MS2DeepScore/cosine comparisons, chemical QC, preliminary results.
2. **Mixed evidence:** no consistent DreaMS advantage across all datasets;
   B has few independent structures beyond Tahlan.
3. **New engineering:** the evidence gate rejects claims that public links
   are enough to justify a strain-resolved benchmark.
4. **Not yet done:** independently validated spectrum→strain→BGC matching,
   NPLinker integration, supervised cross-modal model, full DreaMS article
   replication, or fully compute-matched FL comparison.
5. **Decision to request:** which *existing consortium dataset* contains
   curated sample↔strain, genome↔BGC and positive links, and which
   evaluation criterion matters most?

### Three concrete questions for Joana and Daniel

- Do we have a consortium dataset with sample/strain-level identifiers and
  verified BGC–metabolite links that can be used as a benchmark?
- Should we prioritize robust metabolomic embeddings or a BGC–MS/MS ranking
  baseline once data provenance is established?
- What outcome and hold-out design would make our results useful to PRIORITY?

This is a research *proposal*, not a claim to speak for either researcher.
