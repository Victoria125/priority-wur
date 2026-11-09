# DreaMS embeddings and molecular families

First experiment after the meeting with Joana Tirana and Daniel Probst on 19 October 2026 (embeddings for metabolomics, molecular families, the DreaMS paper).

**Question.** Do embeddings of MS/MS spectra from the pre-trained [DreaMS](https://dreams-docs.readthedocs.io/) model put metabolites of the same molecular family close together? Does DreaMS similarity follow chemical structure better than the modified cosine that GNPS uses, and better than MS2DeepScore? And, as a first look towards federated learning: how different are the labs, and does a small layer trained across labs help?

DreaMS is used as it is (commit `dbec3a0`), on CPU. Only the small layer of Experiment C is trained.

## Data

Six GNPS molecular networking jobs on actinobacteria that the Paired Omics Data Platform links to genomes. `../scripts/download_data.py gnps` keeps them in `../data/raw/gnps/`.

| Run | Dataset | MassIVE (PoDP) | GNPS fragment tolerance |
|---|---|---|---|
| `duncan_salinispora` | *Salinispora*, Duncan et al. 2015 (ion trap) | MSV000079284 | 0.3 Da |
| `crusemann_salstrep` | *Salinispora* + *Streptomyces*, Crüsemann et al. 2017 | MSV000078836 | 0.5 Da |
| `crusemann_streptomyces` | *Streptomyces*, Crüsemann et al. 2017 | MSV000078839 | 0.5 Da |
| `chevrette_insect` | Insect-associated actinobacteria, Chevrette et al. 2019 | MSV000084723 | 0.02 Da |
| `dorrestein_salbus` | *Streptomyces albus* J1074, Dorrestein lab | MSV000079519 | 0.01 Da |
| `tahlan_streptomyces` | Clavulanic acid producing *Streptomyces*, Tahlan lab | MSV000083835 | 0.02 Da |

## Two versions of the input spectra

| Variant | Spectra | Results |
|---|---|---|
| `protocol` (main) | Only spectra that pass DreaMS's own format A check (`DataFormatA().val_spec`), given to every method exactly as GNPS gives them | `runs/<run>/results/`, `federated_prototype/` |
| `all_kept` (sensitivity) | Our cleaning: peaks above the precursor are removed first, then the format A rules. This also keeps spectra that DreaMS's own check rejects: old ion trap data in the Duncan job, and noise above m/z 1000 in every spectrum of the Crüsemann *Streptomyces* job (which has no `protocol` spectra) | `runs/<run>/results_all_kept/`, `federated_prototype_all_kept/` |

Every script takes `--variant protocol` (default) or `--variant all_kept`.

## Steps

| Script | What it does | Output |
|---|---|---|
| `scripts/reproduce_dreams_tutorial.py` | **Step 0**: checks that our DreaMS installation reproduces the numbers saved in the DreaMS networking tutorial | `reproduction/` |
| `scripts/prepare_data.py` | Reads the GNPS job, checks that its files agree, writes both input variants | `runs/<run>/data/`, `runs/<run>/results/run_info.json` |
| `scripts/embed_dreams.py` | One 1024-number DreaMS embedding per spectrum (skipped if already up to date) | `data/dreams_embeddings*.npy` |
| `scripts/embed_ms2deepscore.py` | The same with pre-trained MS2DeepScore (run with `.ms2ds_env`) | `data/ms2deepscore_embeddings*.npy` |
| `scripts/exp_a_families.py` | **Experiment A**: neighbours and a DreaMS network vs the GNPS molecular families, against MS2DeepScore, binned cosine, precursor m/z and random; differences with 95% bootstrap intervals (resampling GNPS families) | `results*/exp_a_*` |
| `scripts/exp_b_structures.py` | **Experiment B**: spectral similarities vs structural similarity (Morgan Tanimoto and MHFP) of library-annotated spectra; differences with 95% intervals (resampling molecules); counts for conclusions only with at least 100 different molecules | `results*/exp_b_*` |
| `scripts/exp_c_federated_adapter.py` | **Experiment C** (prototype): a linear layer on frozen DreaMS embeddings, trained to follow structure; local, centralised, FedAvg and SplitFedV2-like training, one lab per client; performance gap between labs (Tirana et al., AAAI 2026) | `federated_prototype*/` |
| `scripts/exp_d_lab_shift.py` | **Experiment D**: can a classifier tell two labs apart from the embeddings, on all spectra and on molecules both labs measured | `lab_shift/` |
| `scripts/linking_feasibility.py` | Which public PoDP datasets could test DreaMS for genome-metabolome linking (genomes, known links, share of spectra DreaMS accepts) | `linking_feasibility/` |
| `scripts/report.py` | All runs side by side | `RESULTS.md` |
| `scripts/check_pipeline.py` | Automatic checks: unit checks, file consistency, and a regression check against `tests/reference/` | exit code |

`dreams_molecular_families.ipynb` runs everything and has the explanations, limitations and open questions. **Results:** [`RESULTS.md`](RESULTS.md).

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
