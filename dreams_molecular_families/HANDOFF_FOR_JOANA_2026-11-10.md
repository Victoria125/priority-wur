# Research update for Joana Tirana and Daniel Probst
**10 November 2026 — PRIORITY exploratory work / actinobacterial MS/MS representations**

## Scope and current status

I explored public actinobacterial datasets linked in the Paired Omics Data Platform (PoDP), then evaluated pre-trained DreaMS embeddings on GNPS molecular families and on putatively identified chemical structures. I compared DreaMS with MS2DeepScore and spectral cosine baselines and investigated an exploratory multi-client adapter on frozen DreaMS embeddings.

The scientific motivation is to test whether reliable metabolomic representations could help **genome–metabolome linking** in the broader PRIORITY project. This is a research direction, **not yet an implemented or validated BGC–MS/MS linking model**.

## What was corrected in this branch

1. **Library annotation selection.** GNPS can provide multiple library candidates for one spectrum. The original parser retained only the highest-MQScore hit **before** the exact-mass and structural checks. Now all candidate hits are preserved, RDKit/precursor/score/shared-peaks/InChI/adduct checks run on each, and the best *valid* hit is chosen per spectrum. This avoids discarding a valid exact match when a higher-scoring analog hit is invalid. The metadata remains one row per original spectrum, aligned with the MGF and embedding rows.
2. **Fail-closed quality checks.** Missing shared-peaks evidence does not pass by default; missing mass evidence is not accepted. When both measured precursors are present, the directly observed mass difference must pass; GNPS MassDiff may be used only when one reference is missing. For [M+H]+, an actual library precursor m/z must agree with the structure. Invalid nonempty InChI is not treated as absent.
3. **Peak order.** The selected peaks are consistently sorted by m/z, including if the spectrum has fewer than 100 peaks.
4. **Provenance.** A GNPS job with no identifiable spectral inputs can no longer receive an OK result. The fallback GNPS archive used by data preparation is also checked.
5. **Federated terminology and budgets.** The former "SplitFedV2-like" path was not split learning. It is now an explicitly named sequential-client baseline, with 300 **total** optimizer steps. The report and source document that the federated and centralised methods are not fully compute-matched.

## Verification

Run the lightweight regression suite without downloading DreaMS weights:

    python -m pip install -r dreams_molecular_families/tests/requirements-ci.txt
    python -m unittest discover -s dreams_molecular_families/tests -v

The suite checks valid lower-scoring exact matches, invalid analogs, missing chemistry evidence, InChI disagreement, empty result schemas, one-row-per-spectrum alignment, peak ordering and missing-input provenance. A GitHub Actions job is included.

**Important limits:** Only targeted integrity tests are covered; they do not validate GNPS download, actual model inference, long training, scientific conclusions or the entire end-to-end pipeline. There has been no independent re-run of the complete pipeline after this change in the editing environment.

## Re-run required before interpreting scientific results

The existing \`RESULTS.md\` and \`federated_prototype/RESULTS_federated_prototype.md\` are **historical, pre-fix output snapshots**. Since chemical annotations can change, the B, C and D outputs need to be regenerated from the corrected metadata. Some spectral comparisons may also be affected by the peak sorting correction.

Suggested order using the already documented local environments:

1. Run \`scripts/prepare_data.py\` again for each run and both variants. Confirm that the number and identity of accepted independent structures are plausible and that metadata length equals MGF/embedding length.
2. Confirm that DreaMS and MS2DeepScore embeddings correspond to the current spectral inputs, including model versions/checkpoints. Recompute if the inputs/model changed.
3. Re-run Experiments A and B, followed by C and D; regenerate the consolidated results reports. Preserve old results as historic, not post-fix evidence.
4. Validate at least one complete run from a clean environment and inspect the observed per-run coverage, data provenance and annotation attrition.

The earlier snapshot was mixed rather than consistently favorable to DreaMS (e.g. Tahlan B1 Spearman: DreaMS 0.266, MS2DeepScore 0.319). These are **old, pre-fix values**; they should not be quoted as updated results.

## Questions for the next meeting

- Is there a preferred PRIORITY dataset or known BGC–MS/MS link set that we should prioritize?
- Should the immediate next task be **validated metabolomic embeddings**, an **NPLinker linking benchmark**, or training a model for a specific omics layer?
- What success metric and independent validation protocol would be most useful for the team?

A defensible next experiment would use a dataset with known genome–metabolome links, run NPLinker as a baseline, then test whether a DreaMS-based spectral similarity improves link ranking while controlling for structure, strain and dataset leakage.
