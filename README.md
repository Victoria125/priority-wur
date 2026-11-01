# priority-wur

Exploratory groundwork for a possible contribution to **PRIORITY**: *Prioritising novel genomic and chemical space via federated learning and preclinical development* (NWO KIC; Leiden / WUR Bioinformatics, with industrial and international partners). The focus is public paired-omics data for actinobacteria and candidate representations per omic layer, as a basis for later federated-learning experiments.

This is reconnaissance, not a project plan agreed with the PRIORITY team. No data here is synthetic: everything is either downloaded from, or read on demand from, public repositories.

## Contents

| Path | What it is |
|---|---|
| `docs/01_initial_exploration.md` | Short write-up (English) to share: datasets, representation models, candidate tasks, open questions |
| `docs/00_analise_critica.md` | Personal notes (Portuguese): who's who, what PRIORITY is, critique of the initial plan |
| `data/dataset_candidates.csv` | Verified dataset inventory, with accessions, pairing status and caveats |
| `data/representation_models.csv` | Embedding / foundation-model candidates per modality |
| `data/derived/` | Tables produced by the scripts (see below) |
| `scripts/podp_inventory.py` | Downloads all Paired Omics Data Platform (PoDP) projects and flags actinobacterial ones |
| `scripts/massive_files.py` | Summarises a MassIVE dataset's files (sizes, sources, metadata) without downloading it |
| `scripts/download_data.py` | Downloads genomes, spectra and extras; resumable, with a disk guard and a manifest |
| `scripts/show.py` | Short on-screen tables of the results: `datasets`, `links`, `bigset`, `genomes` |
| `priority_data/remote.py` | On-demand access to data too large to keep (local copy first, remote otherwise) |
| `examples/remote_access_demo.py` | Runs every remote-access path against real data |

## What is local and what is remote

| Data | Where | How to use it |
|---|---|---|
| Genomes of the PoDP actinobacteria (312 NCBI assemblies, FASTA + GenBank, ~7 GB) | `data/raw/genomes/<accession>/` | files; mapping in `data/derived/genome_accessions.csv` |
| GNPS molecular-network results registered in PoDP (clustered MS/MS + network, 14 projects incl. MSV000078836/78839, ~150 MB) | `data/raw/gnps/<MSV>/<task>/` | the input NPLinker reads; use these instead of waiting for raw files |
| Raw spectra of the 26 actinobacterial PoDP entries (1,248 mzXML/mzML, ~291 GB) | `data/raw/spectra/<MSV>/...` (downloading, smallest datasets first) | `remote.iter_spectra(acc, path)` reads the local copy, or the remote one until it arrives |
| MIBiG 4.0, GEO processed tables (GSE318666, GSE132487), lydicamycin genomes | `data/raw/mibig`, `data/raw/geo`, `data/raw/genomes` | files |
| **MSV000094769** (1,916 mzML + 359 genomes, 161 GB) | remote | `remote.iter_spectra`, `remote.export_ms2_mgf` (keeps only MS/MS), `remote.genome_fasta` |
| Any single spectrum | remote | `remote.usi_spectrum("mzspec:MSV...:file:scan:N")` (GNPS2 USI service) |
| Any NCBI genome, e.g. NPDC strains | remote, cached on first use | `remote.genome_fasta("GCF_...")` |
| Raw RNA-seq (e.g. XJ54: 15 runs, 30.5 GB) | remote | `remote.rnaseq_runs("PRJNA1397920")`, `remote.iter_fastq(url)` |

Remote files are fetched, parsed and discarded (pyteomics needs a seekable file, so mzML/mzXML pass through a temporary file that is deleted right away).

**Genome matching.** PoDP often gives JGI IDs or free-text strain labels. `download_data.py` resolves each linked genome to an NCBI assembly: directly (187), by BioSample (19) or nucleotide accession (1), by strain ID within the genus (131), or by strain ID under a renamed species (17, flagged `strain_match_species_renamed`, e.g. the 2020 split of *Salinispora pacifica* and *S. albus* J1074 → *S. albidoflavus*). 9 of 364 have no NCBI assembly (JGI-only strains, or a label without a strain).

## Reproduce

```bash
python scripts/podp_inventory.py                       # stdlib only
python -I scripts/download_data.py --dry-run           # sizes
python -I scripts/download_data.py genomes spectra extras
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python examples/remote_access_demo.py
```

MassIVE serves ~1 MB/s per connection and rate-limits above 3, so the full raw download takes on the order of a day or two; 88% of it is two datasets of old profile-mode files (MSV000078836, MSV000078839), whose GNPS results are already local. `download_data.py` resumes where it stopped, stops before free disk space drops below 40 GB (`--min-free-gb`), uses 3 parallel downloads by default (MassIVE rate-limits above that), and appends every attempt to `data/derived/download_manifest.csv`. Raw data, caches and logs are git-ignored. Numbers in the docs come from runs on 11 October 2026.
