# Initial exploration: public data and representations for PRIORITY

*Draft by Vitória Maia, 11 October 2026. This is for discussion, not a proposal.*

## My understanding (please correct me)

- PRIORITY ("Prioritising novel genomic and chemical space via federated learning and preclinical development", NWO KIC) aims to find new antibiotic families in actinomycetes. Partners include NAICONS, Fundación MEDINA, the NPDC at UF Scripps and JGI.
- Federated learning makes sense here because much of the useful paired data sits in (semi-)private strain collections. That is the argument made in Medema & van Wezel, *PLOS Biol* 2025, with MELLODDY as the model.
- Your current bottleneck is choosing (1) a dataset to train on and (2) a representation for each omic layer, with the focus on actinobacteria.

## 1. Public paired data for actinobacteria

I queried the Paired Omics Data Platform (PoDP) API for all 81 projects (`scripts/podp_inventory.py`):

- **26 entries (25 MassIVE accessions) are actinobacteria-dominant.**
- Together they hold **~395 genomes** and **1,297 curated genome↔metabolome links**.
- They also hold **58 BGC↔MS2 links**, the only real ground truth for linking.

| Dataset | Genomes | Links | Why it matters |
|---|---|---|---|
| MSV000078836 Salinispora (Crüsemann 2017) | 120 | 360 (+9 BGC-MS2) | Classic NPLinker benchmark |
| MSV000078839 Streptomyces (same study) | 39 | 351 (+5) | 3 media × 3 extractions |
| MSV000084723 insect-associated actinobacteria (Chevrette 2019) | 120 | 122 | **Bioactivity assays in the paper**: the closest public proxy for genome + metabolome + activity |
| MSV000085159 / 85158 Salinispora (Jensen lab) | 20 / 27 | 240 / 27 | Third lab, different instrument |
| MSV000079284 Salinispora (Duncan 2015) | 28 | 48 (+2) | Pattern-based genome mining reference |
| ~20 small sets (Nocardia, **NAICONS**, **Leiden**, …) | 1–11 each | small | Partner data; Nocardia set has 9 BGC-MS2 links |

These come from different labs, instruments (QTOF, Orbitrap, ion trap), media and genera. That is exactly the non-IID structure a federated simulation needs.

**A caution on MSV000094769** ("LC-MS2 and BGC paired data from 320 actinomycetes", UF/UCSD/CMU). It is the largest deposit I found: 161 GB, 1,916 unique mzML files and 359 RefSeq genomes. However, the deposit contains **no file mapping spectra to genomes**. The mzML files are copied from 15 earlier datasets, one of which is a mixed Populus-isolate collection, and the files are named by plate well. There is no associated paper yet. It is promising, especially if it relates to the NPDC, but the pairing would need to come from the submitters.

**Transcriptomics.** I found no multi-strain actinobacterial dataset with all three layers. The best options are:

- *Saccharomonospora xinjiangensis* XJ54: genome, 15 RNA-seq samples and timsTOF metabolomics, also in PoDP. It is one strain, so it is useful for exploration only.
- The *S. coelicolor* iModulon compendium (454 expression profiles). It is the most realistic basis for a transcriptomic representation.
- GSE132487 (*S. coelicolor* time series, SYSTERACT) and GSE318666 (4-strain Streptomyces community).

Reference resources: NPDC (~18.7k genomes, strains orderable) and MIBiG 4.0 (BGC → compound, partial activity labels).

Full table with accessions and caveats: `data/dataset_candidates.csv`.

## 2. Representation candidates per layer

| Layer | In-house first | External to benchmark |
|---|---|---|
| Genome / BGC | antiSMASH + BiG-SCAPE 2 / BiG-SLiCE features | ESM-2 pooled per BGC; **gLM2** (prokaryote/metagenome-trained, gene-context aware); Bacformer (strain level); Evo 2 later |
| MS/MS spectra | Spec2Vec, MS2DeepScore, MS2Query (matchms) | DreaMS (foundation model) |
| Structures | MAP4 / MHFP | NPClassifier classes |
| Transcriptome | — | iModulons (ICA). No bacterial transcriptomic foundation model exists; Geneformer/scGPT are human single-cell |

Dimitris' talk made me curious whether nucleotide LMs like the ones he used would work on actinobacterial BGCs, which span 10–100+ kb in ~70% GC genomes. Protein-level and gene-context models (ESM-2, gLM2) could be a complementary comparison. His evaluation design would be a good template either way. Details: `data/representation_models.csv`.

## 3. Three possible first tasks (each has labels and a metric)

1. **Link ranking (BGC/GCF ↔ spectrum).** Baseline: NPLinker 2 scores (Metcalf, Rosetta, NPClassScore) on the PoDP actinobacteria sets. Compare an embedding-similarity scorer. Evaluate on the 58 known BGC-MS2 links and MIBiG matches. The ground truth is small, which is itself a finding.
2. **BGC → activity class** (antibacterial / antifungal / cytotoxic). Train on MIBiG and compare the Walker & Clardy 2021 feature baseline with ESM-2/gLM2 embeddings. Split by gene cluster family or genus. This is directly about *prioritising*.
3. **Federated simulation.** Treat each source lab as a client: Moore/Crüsemann, Jensen, Duncan, Currie/Chevrette, NAICONS, Leiden. Compare local vs. centralised vs. FedAvg/FedProx on task 1 or 2 (Flower). Clients naturally differ in genus, instrument and protocol, so this also tests the heterogeneity questions from your split-FL work.

## 4. Open questions

1. Which target matters most for PRIORITY now: BGC↔molecule links, activity prediction, or novelty/dereplication?
2. In the real consortium, will partners hold *different strains with the same modalities* (horizontal FL) or *different modalities for the same strains* (vertical FL, where split learning fits naturally)? This decides how I should build the simulation.
3. Is there a dataset you already work with, such as Leiden or NPDC data or MSV000094769, that I should align with? Or would a curated public benchmark (section 1) be useful?
4. How important is transcriptomics for the first year: is partner RNA-seq planned, or should it come later?
5. Is there a group repository or compute (e.g. WUR HPC) I should use?
6. Have you or the group already tried any of these datasets or models and ruled them out? That would save me from going down the same path.

*All numbers above come from public APIs on 11 Oct 2026; the scripts in `scripts/` reproduce them.*
