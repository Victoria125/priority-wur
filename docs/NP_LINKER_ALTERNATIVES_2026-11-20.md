# Research alternatives to NPLinker: decision memo (20 November 2026)

**Status:** proposed experiments, **not** statements attributed to Joana Tirana
or Daniel Probst. No experimental evidence yet that any alternative is superior
on this PRIORITY dataset. Only the existing embedding benchmarks and the
no-training fixed-fusion prototype are implemented.

## Documented priorities and scope

On 6 October Joana emphasized finding a **good actinobacterial dataset** and
**good embedding/foundation models for each omic layer**. The DreaMS paper
was shared after the 19 October meeting:
https://www.nature.com/articles/s41587-025-02663-3

An explicit, verbatim criticism of NPLinker could not be recovered from the
available meeting excerpts. The user recalls reservations; these should be
clarified as scientific questions rather than attributed quotations.

## Why NPLinker alone is not the default next step

NPLinker is an established modular baseline, not an invalid framework. It
can use co-occurrence and other scoring functions, but its strain labels and
cross-modal associations must be accurate. See:
https://nplinker.github.io/nplinker/latest/quickstart/
https://pmc.ncbi.nlm.nih.gov/articles/PMC8130963/

Current PoDP inventory rows and name-matched GNPS hits are **insufficient
to verify strain-resolved BGC–MS/MS ground truth**. A model switch cannot
fix missing sample-to-strain provenance. See:
`dreams_molecular_families/scripts/linking_evidence.py`.

## Options ranked for the **next research milestone**

| Option | Required data | Feasibility now | Scientific output | Critical caveat |
|---|---|---|---|---|
| 1. Validate pretrained MS/MS representations across datasets and chemical structures | Existing DreaMS/MS2DeepScore embeddings, checked library annotations | **High** | Dataset-level correlations, analog retrieval, uncertainty and domain coverage | Many structural labels are putative; same-lab correlations may not generalize |
| 2. Pre-specified fusion of DreaMS + MS2DeepScore similarity scores (50/50) | Same cached embeddings and compounds | **High** | Does complementary information improve ranking? | Test weights were not trained; a numerical gain is not proof of generalization |
| 3. Out-of-distribution and leave-laboratory-out analysis | Enough **independently matched** chemical structures across labs | **Moderate/limited** | Generalization limits and embedding quality control | Existing 21/16/2/1 structure subsets are small; comparing different chemistries confounds lab effects |
| 4. BGC-class ↔ metabolite-class consistency ranking (without NPLinker) | antiSMASH/BiG-SCAPE BGC annotation; experimentally supported metabolite structures or CANOPUS/NPClassifier class; strain mapping for evaluation | **Later** | Explainable *candidate* links by shared natural-product classes | Broad chemical classes have many false matches, especially without strain evidence |
| 5. NPOmix as an independent linking baseline | Strain-matched genotypes and MS/MS, labels needed for evaluation | **Later** | Direct BGC–MS/MS ranking baseline not relying on NPLinker | Published comparisons are dataset-specific and require an equivalent validation protocol |
| 6. Dual-encoder contrastive BGC embeddings (ESM-2/gLM2) + MS/MS embeddings (DreaMS) | Many independently validated positive BGC–MS2 pairs, reliable hard negatives, held-out families | **Not yet defensible** | New multimodal learned ranking | Too few independent positives now, high leakage/overfitting risk |
| 7. Bioactivity-first ranking | Validated activity assays linked to strain, BGC or compound | **Only after curation** | More directly useful antibiotic discovery objective | Do not equate spectral class with experimentally measured bioactivity |

NPOmix reference:
https://pmc.ncbi.nlm.nih.gov/articles/PMC9802219/

An example of the importance of evaluation differences:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12247221/

## Immediate executable milestones

1. `python dreams_molecular_families/scripts/representation_triage.py`
   to summarize *recorded* model performance by dataset, input coverage,
   number of chemically checked structures, and confidence-interval caveats.
2. `python dreams_molecular_families/scripts/exp_e_fixed_fusion.py --run tahlan_streptomyces`
   to run a genuinely new pre-specified score-level fusion experiment on
   stored embeddings (no training); outputs must be checked against the
   previous DreaMS/MS2DeepScore results on the same structures.
3. `python dreams_molecular_families/scripts/linking_evidence.py --dataset MSV000078836`
   to check if we have sufficient independent provenance. It should say **not
   ready** without manually curated cross-modal identifiers.
4. Avoid creating a federation or multimodal model until an independent
   test set and meaningful target are agreed.

**Important:** independently computed structure annotations and data source
mapping remain scientific judgment calls. Passing unit tests cannot certify
those labels.

## Proposal for the next meeting

"I propose to finish a reproducible, dataset-aware evaluation of DreaMS and
MS2DeepScore for actinobacterial MS/MS data. I also tested a simple fixed
fusion as an exploratory baseline. We then need to decide which sample-to-
strain mapping is reliable enough to support any genome–metabolome ranking.
I would avoid claiming an NPLinker replacement or training a new cross-
modal model before that evidence exists. Which evaluation task and dataset
would best serve PRIORITY?"
