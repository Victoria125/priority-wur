# DreaMS embeddings and molecular families

First experiment after the meeting with Joana Tirana and Daniel Probst on 19 October 2026 (embeddings for metabolomics, molecular families, the DreaMS paper).

**Question.** Do embeddings of MS/MS spectra from the pre-trained [DreaMS](https://dreams-docs.readthedocs.io/) model put metabolites of the same molecular family close together? And does DreaMS similarity follow chemical structure better than the modified cosine that GNPS uses?

No model is trained here: DreaMS is used as it is (commit `dbec3a0`), on CPU.

## Data

Three GNPS molecular networking jobs on actinobacteria that the Paired Omics Data Platform links to genomes (so they can later be connected to BGCs with NPLinker). `scripts/download_data.py gnps` keeps them in `../data/raw/gnps/`.

| Run | Dataset | MassIVE (PoDP) | GNPS job |
|---|---|---|---|
| `duncan_salinispora` | *Salinispora*, Duncan et al. 2015 | MSV000079284 | `c22f44b1…` |
| `crusemann_salstrep` | *Salinispora* + *Streptomyces*, Crüsemann et al. 2017 | MSV000078836 | `9360fa51…` |
| `crusemann_streptomyces` | *Streptomyces*, Crüsemann et al. 2017 | MSV000078839 | `840eadb0…` |

## Steps

| Script | What it does | Output (`runs/<run>/`) |
|---|---|---|
| `scripts/prepare_data.py` | Reads the GNPS job, keeps spectra that follow DreaMS "format A" (the format of its training data) | `data/spectra_clean.mgf`, `data/spectra_metadata.parquet`, `data/gnps_edges.parquet`, `results/00_filter_report.parquet`, `results/run_info.json` |
| `scripts/embed_dreams.py` | One 1024-number DreaMS embedding per spectrum (skipped if already up to date) | `data/dreams_embeddings.npy`, `data/dreams_embeddings.json` |
| `scripts/exp_a_families.py` | **Experiment A**: DreaMS neighbours and a DreaMS network vs the GNPS molecular families, with binned cosine, precursor m/z and random baselines | `results/exp_a_*` |
| `scripts/exp_b_structures.py` | **Experiment B**: DreaMS, modified cosine and binned cosine vs structural similarity (Tanimoto) of library-annotated spectra | `results/exp_b_*` |
| `scripts/report.py` | All runs side by side | `RESULTS.md` |

`dreams_molecular_families.ipynb` runs everything and has the explanations, limitations and open questions. **Results:** [`RESULTS.md`](RESULTS.md).

## Run it

One-time setup, in this folder ([uv](https://docs.astral.sh/uv/) creates the Python 3.11 environment DreaMS needs):

```bash
git clone https://github.com/pluskal-lab/DreaMS.git .DreaMS
git -C .DreaMS checkout dbec3a0b514a99e5056cfccde4559fda8cfe8129
uv venv .dreams_env --python 3.11
uv pip install --python .dreams_env/Scripts/python.exe -e ./.DreaMS ipykernel nbclient
```

Then open the notebook with the `.dreams_env` kernel and run all, or run the scripts one by one:

```bash
.dreams_env/Scripts/python.exe scripts/prepare_data.py --run duncan_salinispora
.dreams_env/Scripts/python.exe scripts/embed_dreams.py --run duncan_salinispora
.dreams_env/Scripts/python.exe scripts/exp_a_families.py --run duncan_salinispora
.dreams_env/Scripts/python.exe scripts/exp_b_structures.py --run duncan_salinispora
.dreams_env/Scripts/python.exe scripts/report.py
```

The first embedding run downloads the DreaMS weights (2.6 GB) from Hugging Face. On a laptop CPU, DreaMS embeds about 4 spectra per second. `.DreaMS/` and `.dreams_env/` are git-ignored; the embeddings and results are in git, so nothing has to be recomputed. `env_versions.txt` lists the exact package versions used.

To add a dataset, add it to `RUNS` in `scripts/common.py` (any GNPS classic molecular networking job) and to `RUNS` in the notebook.
