# Exploratory fixed-fusion result — Tahlan actinobacteria (20 November 2026)

**Status: real cached embeddings, no neural training, not a new biological link.**
The experiment was executed in GitHub Actions on an existing Tahlan
(`tahlan_streptomyces`, `protocol`) dataset, using the originally
recorded DreaMS and MS2DeepScore embeddings and the chemical annotation
QC in `annotations.py`. CI run:
https://github.com/Victoria125/priority-wur/actions/runs/44657070997

## Design

- All 176 selected compounds have a distinct first block of InChIKey,
  after chemical candidate filters. **Structures are still putatively annotated.**
- The 15,400 non-diagonal pairs provide the Spearman comparison between
  learned spectral similarities and RDKit Morgan Tanimoto similarities.
  These pairs are **not 15,400 independent molecules**.
- The new method uses a FIXED 50/50 arithmetic average of the two cosine
  similarity score matrices. No weights or hyperparameters were optimized
  on Tahlan and no new model was trained. The networks themselves are not
  concatenated, retrained or fine-tuned.
- The original model baseline outputs were reproduced to within 0.005
  of their previously reported 0.266 and 0.319 Spearman values.

## Real results

| Method | Spearman with Tanimoto | Top-1 analog Tanimoto |
|---|---:|---:|
| DreaMS (published checkpoint) | 0.266468 | 0.508381 |
| MS2DeepScore (published checkpoint) | 0.319076 | 0.507183 |
| Fixed 50/50 score fusion | **0.340985** | **0.522952** |

These are descriptive point estimates. The fusion does better numerically
here, but does NOT establish general improvement.

## Exploratory paired uncertainty analysis

The Spearman deltas use a *delete-one-molecule jackknife* with a
normal-approximation 95% interval; top-1 deltas use a *query-molecule
bootstrap* holding the candidate library fixed. Both methods require
caution because molecular families, pretrained model overlap, and
library labels may violate simplifying independence assumptions.

| Contrast | Delta Spearman | Approx. interval | Includes zero? |
|---|---:|---:|---|
| Fusion − DreaMS | +0.074516 | [+0.013625, +0.135408] | No |
| Fusion − MS2DeepScore | +0.021909 | [−0.007597, +0.051414] | **Yes** |

| Contrast | Delta top-1 Tanimoto | Approx. interval | Includes zero? |
|---|---:|---:|---|
| Fusion − DreaMS | +0.014571 | [−0.006390, +0.035788] | **Yes** |
| Fusion − MS2DeepScore | +0.015769 | [−0.004700, +0.035067] | **Yes** |

**Critical interpretation:** relative to the stronger baseline,
MS2DeepScore, both exploratory confidence intervals include zero.
Therefore it is NOT defensible to state that fusion is convincingly
better. It is only a modest and interesting numerical signal.

There is no independent cross-lab holdout here. Future preregistered
evaluation should include independent molecular families / external
labs, leakage checks versus published training spectra, instrument
and ionization conditions, and biologically trustworthy annotations.

## Research decision

- **Keep:** DreaMS and MS2DeepScore as nonexclusive representation
  baselines, with provenance-aware chemical QC.
- **Investigate further only if worthwhile:** fixed fusion on an
  independent curated chemical test set. No new model is justified yet.
- **Do not claim:** improved BGC–metabolite linking, validated spectrum–
  strain matching, or that NPLinker is scientifically inferior.
- **Ask Joana/Daniel:** Which curated strain-level dataset and scientific
  outcome should be used next? Is it better to evaluate metabolomics
  similarity or a fully cross-modal task when identifiers are available?

Run reproducibly after checking out the research branch:

```bash
python dreams_molecular_families/scripts/exp_e_fixed_fusion.py \
    --run tahlan_streptomyces --variant protocol \
    --output fusion_tahlan.json
```

The CI report can be downloaded as the `research-review-results`
artifact (JSON outputs and triage Markdown). A passing CI job is a
software verification, not biological ground truth.
