# DreaMS embeddings and molecular families

First experiment after the meeting with Joana Tirana and Daniel Probst on 19 October 2026 (embeddings for metabolomics, molecular families, the DreaMS paper).

**Question.** Do embeddings of MS/MS spectra from the pre-trained [DreaMS](https://dreams-docs.readthedocs.io/) model put metabolites of the same molecular family close together? Does DreaMS similarity follow chemical structure better than the modified cosine that GNPS uses, and better than MS2DeepScore? And, as a first look towards federated learning: how different are the labs, and does a small layer trained across labs help?

DreaMS is used as it is (commit `dbec3a0`), on CPU. Only the small layer of Experiment C is trained.

> **10 Nov 2026 quality-control update:** Chemical annotation selection, m/z sorting, provenance, embedding cache identity and sequential-client training have been corrected in [the QA branch](https://github.com/Victoria125/priority-wur/pull/1). The existing [results tables](RESULTS.md) are **historical pre-fix snapshots**, not re-run post-fix evidence. See [the handoff for Joana and Daniel](HANDOFF_FOR_JOANA_2026-11-10.md).

## Data

Six GNPS molecular networking jobs on actinobacteria that the Paired Omics Data Platform links to genomes. `../scripts/download_data.py gnps` keeps them in `../data/raw/gnps/`.

| Run | Dataset | MassIVE (PoDP) | GNPS fragment tolerance |
|---|---|---|---|
| `duncan_salinispora` | *Salinispora*, Duncan et al. 2015 (ion trap); files read from MSV000079484, the same files as MSV000079284 | MSV000079284 | 0.3 Da |
| `crusemann_salstrep` | *Salinispora* (MSV000078836, uploaded from the submitter's workspace) + *Streptomyces* (MSV000078839), Crüsemann et al. 2017 | MSV000078836 | 0.5 Da |
| `crusemann_streptomyces` | *Streptomyces*, Crüsemann et al. 2017 | MSV000078839 | 0.5 Da |
| `chevrette_insect` | Insect-associated actinobacteria, Chevrette et al. 2019 | MSV000084723 | 0.02 Da |
| `dorrestein_salbus` | *Streptomyces albus* J1074, Dorrestein lab | MSV000079519 | 0.01 Da |
| `tahlan_streptomyces` | Clavulanic acid producing *Streptomyces*, Tahlan lab | MSV000083835 | 0.02 Da |

## Two versions of the input spectra

| Variant | Spectra | Results |
|---|---|---|
| `protocol` (main) | Only spectra that pass DreaMS's own format A check (`DataFormatA().val_spec`), given to every method exactly as GNPS gives them | `runs/<run>/results/`, `federated_prototype/` |
| `all_kept` (sensitivity) | Our cleaning: peaks above the precursor are removed first, then the format A rules. This also keeps spectra that DreaMS's own check rejects: old ion trap data in the Duncan job, and noise above m/z 1000 in every spectrum of the Crüsemann *Streptomyces* job (which has no `protocol` spectra) | `runs/<run>/results_all_kept/`, `federated_prototype_all_kept/` |

**Important:** `all_kept` is *not necessarily a superset* of `protocol`. They are two separately filtered populations: for example, Dorrestein has 2,392 protocol spectra but 2,053 all_kept spectra in the historical snapshot. Compare their intersection and composition before attributing differences to model robustness.

Every script takes `--variant protocol` (default) or `--variant all_kept`. Files about the run itself (`run_info.json`, `input_report.parquet`, `provenance.json`) are in `runs/<run>/`, outside both results folders.

## Checks built into the pipeline

- **Provenance.** `check_provenance.py` reads the input files of each GNPS job (`upload_file_mapping` in params.xml), groups the spectra files by source and checks them against the documented MassIVE datasets (`RUNS[run]["sources"]`), including that the file names exist in those datasets.
- **Charge.** The charge GNPS gives is kept in the MGF files (no `CHARGE` line when it is unknown) and counted in `run_info.json`. Embeddings are matched to their spectra by a fingerprint of cluster ids, precursors and peaks (`common.spectra_fingerprint`), which the charge line does not change.
- **Annotations.** `annotations.py`: GNPS library candidates are chemically screened **before** the best valid match is chosen per spectrum (so a higher-scoring analog cannot hide a valid exact match). Valid SMILES, exact mass, MQScore >= 0.7 and >= 6 shared peaks, InChI agreement, and [M+H]+ mass consistency are required. No quality field defaults to a passing value. Experiments B/C/D then deduplicate by compound; MGF/embedding metadata remain one row per spectrum.
- **Equivalent inputs.** All methods get the same 100 most intense peaks that DreaMS's embedding model keeps (`common.top_peaks`), sorted by ascending m/z for spectral similarity code. Cached embeddings are additionally checked against the recorded model/revision/peak settings when applicable.
- **Automatic checks.** `check_pipeline.py` fails when required data or results are missing; the only allowed absence is a variant that `run_info.json` declares empty.

## Steps

| Script | What it does | Output |
|---|---|---|
| `scripts/reproduce_dreams_tutorial.py` | **Step 0**: checks that our DreaMS installation reproduces the numbers saved in the DreaMS networking tutorial | `reproduction/` |
| `scripts/prepare_data.py` | Reads the GNPS job, checks that its files agree, writes both input variants with the real charges | `runs/<run>/data/`, `runs/<run>/run_info.json`, `runs/<run>/input_report.parquet` |
| `scripts/check_provenance.py` | Where the spectra of each GNPS job come from, against the documented MassIVE datasets | `runs/<run>/provenance.json`, `provenance/` |
| `scripts/embed_dreams.py` | One 1024-number DreaMS embedding per spectrum (skipped if already up to date) | `data/dreams_embeddings*.npy` |
| `scripts/embed_ms2deepscore.py` | The same with pre-trained MS2DeepScore (run with `.ms2ds_env`) | `data/ms2deepscore_embeddings*.npy` |
| `scripts/exp_a_families.py` | **Experiment A**: neighbours and a DreaMS network vs the GNPS molecular families, against MS2DeepScore, binned cosine, precursor m/z and random; differences with 95% bootstrap intervals (resampling GNPS families) | `results*/exp_a_*` |
| `scripts/exp_b_structures.py` | **Experiment B**: spectral similarities vs structural similarity (Morgan Tanimoto and MHFP) of validated, independent compounds; differences with 95% intervals (resampling molecules); counts for conclusions only with at least 100 compounds | `results*/exp_b_*` |
| `scripts/exp_c_federated_adapter.py` | **Experiment C**: a linear layer on frozen DreaMS embeddings; 5 molecule splits; lab, IID and Dirichlet client partitions; pre-trained, local, centralised, FedAvg, FedProx and a **sequential-client baseline (not SplitFed)**, with validation-based selection and paired 95% intervals; methods are not fully compute-matched | `federated_prototype*/` |
| `scripts/exp_d_lab_shift.py` | **Experiment D**: can a classifier tell two labs apart from the embeddings, on all spectra and on molecules both labs measured | `lab_shift/` |
| `scripts/linking_feasibility.py` | Which public PoDP datasets could test DreaMS for genome-metabolome linking (genomes, known links, share of spectra DreaMS accepts) | `linking_feasibility/` |
| `scripts/report.py` | All runs side by side | `RESULTS.md` |
| `scripts/check_pipeline.py` | Automatic checks: unit checks, file consistency, and a regression check against `tests/reference/` | exit code |

`dreams_molecular_families.ipynb` runs everything and has the explanations, limitations and open questions. **Historical (pre-fix) results:** [`RESULTS.md`](RESULTS.md).

## Fast regression tests (no DreaMS weights required)

From the repository root:

```bash
python -m pip install -r dreams_molecular_families/tests/requirements-ci.txt
python -m unittest discover -s dreams_molecular_families/tests -v
```

These cover annotation quality, multiple hits, empty matches, spectrum-row alignment, peak sorting, provenance, and cache identity. GitHub Actions runs the tests on the QA branch and pull request. **They do not replace a clean end-to-end run.** To regenerate the scientific tables, re-run `prepare_data.py` for all relevant datasets followed by Experiments A/B/C/D, model comparisons, and `report.py` in the project's documented environments. The corrected source has not been used to regenerate the historical results automatically.

## Training a DreaMS model from scratch (GPU, Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Victoria125/priority-wur/blob/dreams-molecular-families/dreams_molecular_families/colab_train_from_scratch.ipynb)

`colab_train_from_scratch.ipynb` trains a smaller DreaMS model from zero on one Colab GPU (self-supervised, as in the DreaMS paper) and compares it with the published DreaMS on our data.

| Script | What it does |
|---|---|
| `scripts/train_from_scratch.py` | DreaMS pre-training with DreaMS's own model and data classes and the settings of `pre_train.sh`, but smaller (`--size tiny/small/published`), on one device, with a time limit, saving every 500 steps and continuing from the last save. It works around a bug in DreaMS commit `dbec3a0`: `to_hot()` asks `to_classes()` for two values, but that return line is commented out, so the published pre-training code stops with "too many values to unpack" |
| `scripts/embed_ssl.py` | Embeddings of a self-supervised model (precursor token, last layer): the published DreaMS before fine-tuning (`--name dreams_ssl`) or ours (`--ckpt`) |
| `scripts/compare_models.py` | Experiments A1, B1 and B2 for our model vs the published DreaMS, with 95% intervals |

Training data: random samples of GeMS from the DreaMS authors (`GeMS_A1_DreaMS_rand500k.hdf5`, 2.2 GB; `rand50k`, 0.2 GB for tests), from Hugging Face `roman-bushuiev/GeMS`. The pipeline was tested on a laptop CPU with a tiny model (training, resuming, embeddings, comparison); the Colab run itself has not been tested yet.

## Run it

One-time setup, in this folder ([uv](https://docs.astral.sh/uv/) creates the Python 3.11 environments). DreaMS and MS2DeepScore pin different package versions, so each has its own environment:

```bash
git clone https://github.com/pluskal-lab/DreaMS.git .DreaMS
git -C .DreaMS checkout dbec3a0b514a99e5056cfccde4559fda8cfe8129
uv venv .dreams_env --python 3.11
uv pip install --python .dreams_env/Scripts/python.exe -e ./.DreaMS mhfp==1.9.6 ipykernel nbclient

uv venv .ms2ds_env --python 3.11
uv pip install --python .ms2ds_env/Scripts/python.exe "ms2deepscore>=2.6" onnxruntime
```

Then, for each run (`--variant all_kept` for the sensitivity analysis):

```bash
.dreams_env/Scripts/python.exe scripts/reproduce_dreams_tutorial.py      # once
.dreams_env/Scripts/python.exe scripts/prepare_data.py --run tahlan_streptomyces
.dreams_env/Scripts/python.exe scripts/embed_dreams.py --run tahlan_streptomyces
.ms2ds_env/Scripts/python.exe  scripts/embed_ms2deepscore.py --run tahlan_streptomyces
.dreams_env/Scripts/python.exe scripts/exp_a_families.py --run tahlan_streptomyces
.dreams_env/Scripts/python.exe scripts/exp_b_structures.py --run tahlan_streptomyces
```

and then, over all runs: `exp_c_federated_adapter.py`, `exp_d_lab_shift.py`, `linking_feasibility.py`, `report.py` and `check_pipeline.py`.

The first runs download the DreaMS weights (2.6 GB, Hugging Face) and the MS2DeepScore model (416 MB, Zenodo). On a laptop CPU with 8 GB of RAM, DreaMS embeds about 2 to 4 spectra per second; close other programs while it runs. `.DreaMS/`, `.dreams_env/`, `.ms2ds_env/` and `.ms2deepscore/` are git-ignored; the embeddings and results are in git, so nothing has to be recomputed. `env_versions.txt` lists the exact package versions of the DreaMS environment.

To add a dataset, add it to `RUNS` in `scripts/common.py` (any GNPS classic molecular networking job) and to `RUNS` in the notebook.
