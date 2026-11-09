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


def validated_annotations(meta: pd.DataFrame):
    """Rows of meta whose library hit passes the checks, one per molecule.

    Returns (selected, counts): selected has the columns row (position in meta),
    key, adduct, library_score and mol; counts is a list of (step, spectra left).
    """
    hits = meta[meta["smiles"].notna()].copy() if "smiles" in meta else meta.iloc[0:0].copy()
    counts = [("library hits with a SMILES", len(hits))]
    hits["mol"] = [molecule(s) for s in hits["smiles"]]
    hits = hits[hits["mol"].notna()]
    hits["key"] = [structure_key(m) for m in hits["mol"]]
    hits = hits[hits["key"] != ""]
    counts.append(("SMILES read by RDKit", len(hits)))

    if "mass_diff" in hits:
        hits = hits[hits["mass_diff"].abs() <= MAX_MASS_DIFF]
    counts.append((f"exact-mass hits (|query - library precursor| <= {MAX_MASS_DIFF} Da)", len(hits)))

    score = pd.to_numeric(hits["library_score"], errors="coerce")
    peaks = pd.to_numeric(hits.get("shared_peaks", pd.Series(np.inf, index=hits.index)), errors="coerce")
    hits = hits[(score >= MIN_LIBRARY_SCORE) & (peaks >= MIN_SHARED_PEAKS)]
    counts.append((f"MQScore >= {MIN_LIBRARY_SCORE} and >= {MIN_SHARED_PEAKS} shared peaks", len(hits)))

    from_inchi = [inchi_key(i) for i in hits.get("inchi", pd.Series("", index=hits.index))]
    agree = [not k or k == key for k, key in zip(from_inchi, hits["key"])]
    hits = hits[agree]
    counts.append(("InChI agrees with SMILES (when an InChI is given)", len(hits)))

    hits["adduct_norm"] = [normalise_adduct(a) for a in hits.get("adduct", pd.Series("", index=hits.index))]
    is_mh = hits["adduct_norm"] == "M+H"
    reference_mz = pd.to_numeric(hits.get("library_mz", hits["precursor_mz"]), errors="coerce").fillna(hits["precursor_mz"])
    expected = np.array([Descriptors.ExactMolWt(m) + PROTON for m in hits["mol"]]) if len(hits) else np.array([])
    mass_ok = ~is_mh.to_numpy() | (np.abs(reference_mz.to_numpy() - expected) <= MAX_MH_ERROR)
    hits = hits[mass_ok]
    counts.append((f"[M+H]+ hits: library precursor agrees with the structure (<= {MAX_MH_ERROR} Da)", len(hits)))

    hits["is_mh"] = hits["adduct_norm"] == "M+H"
    hits["score_num"] = pd.to_numeric(hits["library_score"], errors="coerce")
    hits = (hits.reset_index().rename(columns={"index": "meta_index"})
            .sort_values(["key", "is_mh", "score_num"], ascending=[True, False, False])
            .drop_duplicates("key"))
    counts.append(("independent compounds (one spectrum per molecule)", len(hits)))

    selected = pd.DataFrame({
        "row": [meta.index.get_loc(i) for i in hits["meta_index"]],
        "key": hits["key"].to_numpy(),
        "adduct": hits["adduct_norm"].to_numpy(),
        "library_score": hits["score_num"].to_numpy(),
        "mol": hits["mol"].to_numpy(),
    }).sort_values("row").reset_index(drop=True)
    return selected, counts
