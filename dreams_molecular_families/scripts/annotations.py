"""Chemical validation of the GNPS library annotations.

A library hit is used as the structure of a spectrum only if it passes all
of these checks, in this order:

1. The SMILES can be read by RDKit. We keep the largest fragment (no salts)
   and neutralise it.
2. Exact-mass hit: the query and library precursor m/z differ by at most
   0.05 Da. This removes analog-search hits, which are other molecules (the
   Duncan job ran with ANALOG_SEARCH=1).
3. A good spectral match: MQScore >= 0.7 and at least 6 shared peaks
   (the GNPS defaults for library matches).
4. If the hit also has an InChI, its structure (first InChIKey block) is the
   same as the SMILES.
5. Adduct names are normalised (M+H, [M+H], [M+H]+ are the same). For [M+H]+
   hits, the library precursor m/z must agree with the structure: neutral
   monoisotopic mass + 1.007276 within 0.05 Da.
6. One spectrum per molecule (first InChIKey block): the [M+H]+ spectrum if
   there is one, then the best match. Metrics then compare independent
   compounds, not the same molecule measured several times or as other adducts.
"""
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

MAX_MASS_DIFF = 0.05      # Da, query vs library precursor (exact-mass hit)
MIN_LIBRARY_SCORE = 0.7
MIN_SHARED_PEAKS = 6
MAX_MH_ERROR = 0.05       # Da, [M+H]+ library precursor vs structure
PROTON = 1.007276
ANNOTATION_QC_VERSION = "2026-11-10-candidate-selection-v2"


def molecule(smiles):
    """SMILES -> neutral RDKit molecule without salts / counter ions (None if invalid)."""
    if not isinstance(smiles, str) or not smiles.strip() or smiles.strip().upper() in ("N/A", "NA"):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    try:
        return rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return mol


def normalise_adduct(adduct) -> str:
    """'[M+H]+', 'M+H', '[M+H]' -> 'M+H'. Unknown or empty -> ''."""
    if not isinstance(adduct, str):
        return ""
    a = adduct.strip().replace(" ", "")
    a = re.sub(r"^\[(.*)\]\d*[+-]*$", r"\1", a)
    return a.rstrip("+-")


def structure_key(mol) -> str:
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else ""


def inchi_key(inchi) -> str:
    if not isinstance(inchi, str) or not inchi.strip().startswith("InChI="):
        return ""
    try:
        mol = Chem.MolFromInchi(inchi.strip())
    except Exception:
        return ""
    return structure_key(mol) if mol is not None else ""


def screen_library_hits(meta: pd.DataFrame):
    """Validate every GNPS candidate hit without deduplicating molecules or spectra.

    Selection MUST be done after these filters: a high-scoring analog or invalid
    structure must not hide a lower-scoring, chemically valid exact-mass hit.
    Missing quality/mass information never counts as evidence of a valid hit.
    Returns (valid_candidates, counts).
    """
    hits = meta.loc[meta["smiles"].notna()].copy() if "smiles" in meta else meta.iloc[:0].copy()
    counts = [("library hits with a SMILES", len(hits))]

    smiles = hits["smiles"] if "smiles" in hits else []
    hits["mol"] = pd.Series([molecule(s) for s in smiles], index=hits.index, dtype=object)
    hits = hits.loc[hits["mol"].notna()].copy()
    hits["key"] = [structure_key(m) for m in hits["mol"]]
    hits = hits.loc[hits["key"] != ""].copy()
    counts.append(("SMILES read by RDKit", len(hits)))

    # Compare the actual precursors when both are present. GNPS MassDiff is
    # accepted as a fallback only when a library precursor is unavailable.
    # If both measurements exist, they must BOTH pass the exact-mass rule.
    reported = pd.to_numeric(
        hits["mass_diff"] if "mass_diff" in hits else pd.Series(np.nan, index=hits.index),
        errors="coerce",
    ).abs()
    query_mz = pd.to_numeric(
        hits["precursor_mz"] if "precursor_mz" in hits else pd.Series(np.nan, index=hits.index),
        errors="coerce",
    )
    library_mz = pd.to_numeric(
        hits["library_mz"] if "library_mz" in hits else pd.Series(np.nan, index=hits.index),
        errors="coerce",
    )
    directly_measured = query_mz.notna() & library_mz.notna()
    direct_diff = (query_mz - library_mz).abs()
    mass_ok = (
        query_mz.notna()
        & ((directly_measured & direct_diff.le(MAX_MASS_DIFF))
           | (~directly_measured & reported.le(MAX_MASS_DIFF)))
        & (reported.isna() | reported.le(MAX_MASS_DIFF))
    )
    hits = hits.loc[mass_ok].copy()
    counts.append((f"exact-mass hits (|query - library precursor| <= {MAX_MASS_DIFF} Da)", len(hits)))

    score = pd.to_numeric(
        hits["library_score"] if "library_score" in hits else pd.Series(np.nan, index=hits.index),
        errors="coerce",
    )
    peaks = pd.to_numeric(
        hits["shared_peaks"] if "shared_peaks" in hits else pd.Series(np.nan, index=hits.index),
        errors="coerce",
    )
    hits = hits.loc[(score >= MIN_LIBRARY_SCORE) & (peaks >= MIN_SHARED_PEAKS)].copy()
    counts.append((f"MQScore >= {MIN_LIBRARY_SCORE} and >= {MIN_SHARED_PEAKS} shared peaks", len(hits)))

    inchis = hits["inchi"].fillna("") if "inchi" in hits else pd.Series("", index=hits.index)
    agree = [
        (not isinstance(i, str) or not i.strip()) or inchi_key(i) == key
        for i, key in zip(inchis, hits["key"])
    ]
    hits = hits.loc[pd.Series(agree, index=hits.index, dtype=bool)].copy()
    counts.append(("InChI agrees with SMILES (when an InChI is given)", len(hits)))

    adducts = hits["adduct"] if "adduct" in hits else pd.Series("", index=hits.index)
    hits["adduct_norm"] = [normalise_adduct(a) for a in adducts]
    is_mh = hits["adduct_norm"] == "M+H"
    reference_mz = pd.to_numeric(
        hits["library_mz"] if "library_mz" in hits else pd.Series(np.nan, index=hits.index),
        errors="coerce",
    )
    expected = pd.Series(
        [Descriptors.ExactMolWt(m) + PROTON for m in hits["mol"]],
        index=hits.index, dtype=float,
    )
    # Missing library_mz is NOT an automatic pass for [M+H]+.
    mass_ok = ~is_mh | (reference_mz - expected).abs().le(MAX_MH_ERROR)
    hits = hits.loc[mass_ok].copy()
    counts.append((f"[M+H]+ hits: library precursor agrees with the structure (<= {MAX_MH_ERROR} Da)", len(hits)))

    hits["is_mh"] = hits["adduct_norm"] == "M+H"
    hits["score_num"] = pd.to_numeric(hits["library_score"], errors="coerce")
    return hits, counts


def best_library_hits(spectra_meta: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """Best *validated* candidate per spectrum; metadata stays one-to-one.

    Called during prepare_data before merging the library into the metadata.
    Unvalidated hits remain available in the original GNPS archive.
    """
    if library.empty:
        return library.copy()
    candidates = spectra_meta[["cluster_index", "precursor_mz"]].merge(
        library, on="cluster_index", how="inner", validate="one_to_many"
    )
    valid, _ = screen_library_hits(candidates)
    if valid.empty:
        return library.iloc[:0].copy()
    best = (valid.sort_values(
        ["cluster_index", "score_num"], ascending=[True, False], kind="stable"
    ).drop_duplicates("cluster_index", keep="first"))
    return best[list(library.columns)].reset_index(drop=True)


def validated_annotations(meta: pd.DataFrame):
    """One spectrum per independent structure after strict chemical validation.

    Returns (selected, counts): selected has row (position in meta), key, adduct,
    library_score, mol; counts is a list of (step, spectra left).
    """
    hits, counts = screen_library_hits(meta)
    if hits.empty:
        counts.append(("independent compounds (one spectrum per molecule)", 0))
        return pd.DataFrame(columns=["row", "key", "adduct", "library_score", "mol"]), counts

    hits = (hits.reset_index().rename(columns={"index": "meta_index"})
            .sort_values(["key", "is_mh", "score_num"],
                         ascending=[True, False, False], kind="stable")
            .drop_duplicates("key", keep="first"))
    counts.append(("independent compounds (one spectrum per molecule)", len(hits)))
    selected = pd.DataFrame({
        "row": [meta.index.get_loc(i) for i in hits["meta_index"]],
        "key": hits["key"].to_numpy(),
        "adduct": hits["adduct_norm"].to_numpy(),
        "library_score": hits["score_num"].to_numpy(),
        "mol": hits["mol"].to_numpy(),
    }).sort_values("row").reset_index(drop=True)
    return selected, counts
