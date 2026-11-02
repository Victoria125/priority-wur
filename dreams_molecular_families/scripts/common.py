"""Shared helpers for the DreaMS + molecular families experiment.

Only numpy, pandas and the Python standard library are used here,
so the code is easy to read and to test.

Every dataset is a "run". Each run has its own folder:
  runs/<run>/data/     inputs for DreaMS and the embeddings
  runs/<run>/results/  tables (Parquet), figures and run_info.json
"""
from __future__ import annotations

import hashlib
import json
import re
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths. They do not depend on the folder you run the scripts from.
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]   # dreams_molecular_families/
REPO_DIR = EXPERIMENT_DIR.parent                       # priority-wur/
RUNS_DIR = EXPERIMENT_DIR / "runs"
LOCAL_GNPS_DIR = REPO_DIR / "data" / "raw" / "gnps"    # filled by scripts/download_data.py

# DreaMS version used for the embeddings (27 Apr 2026).
DREAMS_COMMIT = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"

# ---------------------------------------------------------------------------
# Datasets. All three are GNPS (classic) molecular networking jobs registered
# in the Paired Omics Data Platform (PoDP), so they can later be connected to
# the genomes (BGCs) with NPLinker. scripts/download_data.py already keeps a
# local copy of each job in data/raw/gnps/<MassIVE>/<task>/.
# ---------------------------------------------------------------------------
RUNS = {
    "duncan_salinispora": {
        "task_id": "c22f44b14a3d450eb836d607cb9521bb",
        "massive": "MSV000079284",
        "label": "Salinispora, Duncan et al. 2015",
        # A copy of this job is stored in the NPLinker GitHub repository (it is
        # the example job of the NPLinker docs). The link is pinned to one commit.
        "fallback_url": (
            "https://raw.githubusercontent.com/NPLinker/nplinker/"
            "33400aea0a1eacf05ead24bee9926013874baa95/tests/unit/data/gnps/"
            "ProteoSAFe-METABOLOMICS-SNETS-c22f44b1-download_clustered_spectra.zip"
        ),
        "fallback_md5": "ee61783601039f4a85b8c0df7467e850",
    },
    "crusemann_salstrep": {
        "task_id": "9360fa514804487a9d39b7e7d7e6d514",
        "massive": "MSV000078836",
        "label": "Salinispora + Streptomyces, Crüsemann et al. 2017",
    },
    "crusemann_streptomyces": {
        "task_id": "840eadb054f74d74ae42152a08ed9628",
        "massive": "MSV000078839",
        "label": "Streptomyces, Crüsemann et al. 2017",
    },
}

# For any other GNPS classic job (same URLs that NPLinker uses).
GNPS_URLS = [
    "https://gnps.ucsd.edu/ProteoSAFe/DownloadResult?task={}&view=download_clustered_spectra",
    "https://gnps.ucsd.edu/ProteoSAFe/DownloadResult?task={}&view=download_cytoscape_data",
]

# Where each file lives inside a GNPS job archive. This covers
# METABOLOMICS-SNETS, METABOLOMICS-SNETS-V2 and FEATURE-BASED-MOLECULAR-NETWORKING.
ARCHIVE_PATTERNS = {
    "spectra": [r"^METABOLOMICS-SNETS[^/]*\.mgf$", r"^spectra/specs_ms\.mgf$"],
    "nodes": [
        r"^clusterinfosummarygroup_attributes_withIDs_withcomponentID/[^/]+\.(tsv|clustersummary)$",
        r"^clusterinfo_summary/[^/]+\.tsv$",
    ],
    "edges": [r"^networkedges_selfloop/[^/]+$"],
    "library": [r"^result_specnets_DB/[^/]+\.tsv$", r"^DB_result/[^/]+\.tsv$"],
    "params": [r"^params\.xml$"],
}

# Quality rules copied from DreaMS "format A" (dreams/utils/dformats.py).
# Format A is the data format of GeMS-A, the data DreaMS was pre-trained on.
MAX_PRECURSOR_MZ = 1000.0
MAX_FRAGMENT_MZ = 1000.0
MAX_PEAKS = 128                 # we keep the 128 most intense peaks (DreaMS uses the top 100)
HIGH_PEAK_THRESHOLD = 0.1       # "high" peak = relative intensity above 10%
MIN_HIGH_PEAKS = 3              # at least 3 high peaks
MIN_INTENSITY_AMPLITUDE = 20.0  # max intensity / min intensity

# Binned spectrum baseline: 1 Da bins up to m/z 1000, square-root intensities.
BIN_WIDTH = 1.0


def run_paths(run: str) -> SimpleNamespace:
    """All files of one run."""
    root = RUNS_DIR / run
    data, results = root / "data", root / "results"
    return SimpleNamespace(
        name=run,
        root=root,
        data=data,
        results=results,
        gnps=data / "gnps",
        clean_mgf=data / "spectra_clean.mgf",
        metadata=data / "spectra_metadata.parquet",
        edges=data / "gnps_edges.parquet",
        embeddings=data / "dreams_embeddings.npy",
        embeddings_info=data / "dreams_embeddings.json",
        run_info=results / "run_info.json",
    )


def local_archive(run: str) -> Path | None:
    """The GNPS job archive downloaded by scripts/download_data.py, if present."""
    spec = RUNS.get(run)
    if spec is None:
        return None
    path = LOCAL_GNPS_DIR / spec["massive"] / spec["task_id"] / "download_clustered_spectra.zip"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Download and archive helpers
# ---------------------------------------------------------------------------
def md5sum(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _urlopen(request):
    """Open a URL with the system certificates; if that fails (it happens on some
    laptops), try again with the certificates from the certifi package."""
    try:
        return urllib.request.urlopen(request, timeout=900)
    except urllib.error.URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" not in str(error):
            raise
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        return urllib.request.urlopen(request, timeout=900, context=context)


def download(url: str, dest, post: bool = False) -> Path:
    """Download a file. GNPS classic needs a POST request, GitHub a GET."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, data=b"" if post else None, headers={"User-Agent": "Mozilla/5.0"}
    )
    with _urlopen(request) as response, open(dest, "wb") as f:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def find_archive_members(zip_path) -> dict:
    """Return {role: file name inside the zip} for the files we need."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    found = {}
    for role, patterns in ARCHIVE_PATTERNS.items():
        for pattern in patterns:
            hits = sorted(n for n in names if re.match(pattern, n))
            if hits:
                found[role] = hits[0]
                break
    return found


def is_gnps_archive(zip_path) -> bool:
    try:
        members = find_archive_members(zip_path)
    except zipfile.BadZipFile:
        return False
    return all(role in members for role in ("spectra", "nodes", "edges"))


def extract_gnps_archive(zip_path, out_dir) -> dict:
    """Extract the spectra, nodes, edges, library hits and params files."""
    members = find_archive_members(zip_path)
    missing = [r for r in ("spectra", "nodes", "edges") if r not in members]
    if missing:
        raise ValueError(
            f"{zip_path} does not look like a GNPS molecular networking job "
            f"(missing: {', '.join(missing)})."
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with zipfile.ZipFile(zip_path) as z:
        for role, member in members.items():
            target = out_dir / f"{role}{Path(member).suffix or '.txt'}"
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            paths[role] = target
    return paths


def read_gnps_params(params_path) -> dict:
    """Read a few useful settings from the GNPS params.xml file."""
    if params_path is None or not Path(params_path).exists():
        return {}
    text = Path(params_path).read_text(errors="ignore")
    keys = [
        "task", "workflow", "desc", "reanalyzed_datasets",
        "tolerance.Ion_tolerance", "tolerance.PM_tolerance",
        "PAIRS_MIN_COSINE", "MIN_MATCHED_PEAKS", "TOPK", "MAXIMUM_COMPONENT_SIZE",
    ]
    out = {}
    for key in keys:
        match = re.search(rf'<parameter name="{re.escape(key)}">([^<]*)<', text)
        if match:
            out[key] = match.group(1).strip()
    return out


# ---------------------------------------------------------------------------
# MGF reading and writing
# ---------------------------------------------------------------------------
def parse_charge(value) -> int:
    """'1+' -> 1, '2+' -> 2, '1-' -> -1, '0' or '' -> 0 (unknown)."""
    value = str(value).strip()
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        return 0
    sign = -1 if "-" in value else 1
    return sign * int(digits)


def read_mgf(path) -> list:
    """Read an MGF file into a list of dicts (simple and fast).

    Each dict has: scans (int), precursor_mz (float), charge (int, 0 = unknown),
    mz (np.ndarray) and intensity (np.ndarray).
    """
    spectra, meta, peaks = [], {}, []
    with open(path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                meta, peaks = {}, []
            elif line == "END IONS":
                arr = np.array(peaks, dtype=float).reshape(-1, 2)
                pepmass = meta.get("PEPMASS", "nan").split()[0]
                spectra.append({
                    "scans": int(float(meta.get("SCANS", -1))),
                    "precursor_mz": float(pepmass),
                    "charge": parse_charge(meta.get("CHARGE", "")),
                    "mz": arr[:, 0],
                    "intensity": arr[:, 1],
                })
            elif "=" in line and line[0].isalpha():
                key, value = line.split("=", 1)
                meta[key.strip().upper()] = value.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    peaks.append((float(parts[0]), float(parts[1])))
    return spectra


def write_mgf(spectra, path) -> None:
    """Write spectra in a simple MGF file that DreaMS can read. Order is kept."""
    with open(path, "w", newline="\n") as f:
        for s in spectra:
            f.write("BEGIN IONS\n")
            f.write(f"TITLE=cluster_{s['scans']}\n")
            f.write(f"SCANS={s['scans']}\n")
            f.write(f"PEPMASS={s['precursor_mz']:.5f}\n")
            f.write("CHARGE=1+\n")
            for mz, intensity in zip(s["mz"], s["intensity"]):
                f.write(f"{mz:.5f} {intensity:.6g}\n")
            f.write("END IONS\n\n")


def clean_and_check(spec):
    """Clean one spectrum and check it against the DreaMS format A rules.

    Returns (cleaned_spectrum, None) if it passes, or (None, reason) if not.
    """
    if len(spec["mz"]) == 0:
        return None, "empty spectrum (no peaks at all)"
    if spec["charge"] not in (0, 1):  # 0 = unknown charge, DreaMS assumes +1
        return None, "charge is not +1"
    pmz = spec["precursor_mz"]
    if not np.isfinite(pmz) or pmz > MAX_PRECURSOR_MZ:
        return None, "precursor m/z above 1000"
    mz, intensity = spec["mz"], spec["intensity"]
    # remove empty peaks and peaks above the precursor (noise)
    keep = (intensity > 0) & (mz <= pmz + 1.0)
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) == 0:
        return None, "no peaks below the precursor m/z (looks like noise)"
    if len(mz) > MAX_PEAKS:
        top = np.argsort(intensity)[-MAX_PEAKS:]
        mz, intensity = mz[top], intensity[top]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    if mz.max() > MAX_FRAGMENT_MZ:
        return None, "fragment m/z above 1000"
    relative = intensity / intensity.max()
    if (relative > HIGH_PEAK_THRESHOLD).sum() < MIN_HIGH_PEAKS:
        return None, "fewer than 3 peaks above 10% intensity"
    if intensity.max() / intensity.min() < MIN_INTENSITY_AMPLITUDE:
        return None, "intensity range too small (max/min < 20)"
    return dict(spec, mz=mz, intensity=intensity), None


# ---------------------------------------------------------------------------
# GNPS tables
# ---------------------------------------------------------------------------
def read_nodes(path) -> pd.DataFrame:
    nodes = pd.read_csv(path, sep="\t", low_memory=False)
    nodes = nodes.rename(columns={"cluster index": "cluster_index", "componentindex": "gnps_family"})
    return nodes[["cluster_index", "gnps_family"]]


def read_edges(path) -> pd.DataFrame:
    edges = pd.read_csv(path, sep="\t", low_memory=False)
    edges = edges.rename(columns={"CLUSTERID1": "id1", "CLUSTERID2": "id2", "Cosine": "gnps_cosine"})
    edges = edges[edges["id1"] != edges["id2"]]  # remove self loops
    return edges[["id1", "id2", "gnps_cosine"]].reset_index(drop=True)


def read_library_hits(path) -> pd.DataFrame:
    columns = ["cluster_index", "compound_name", "smiles", "inchi", "library_score", "shared_peaks"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=columns)
    lib = pd.read_csv(path, sep="\t", low_memory=False)
    lib = lib.rename(columns={
        "#Scan#": "cluster_index", "Compound_Name": "compound_name", "Smiles": "smiles",
        "INCHI": "inchi", "MQScore": "library_score", "SharedPeaks": "shared_peaks",
    })
    lib = lib.sort_values("library_score", ascending=False).drop_duplicates("cluster_index")
    return lib[[c for c in columns if c in lib.columns]]


# ---------------------------------------------------------------------------
# Small helpers for the experiments
# ---------------------------------------------------------------------------
def load_metadata_and_embeddings(p: SimpleNamespace):
    meta = pd.read_parquet(p.metadata)
    emb = np.load(p.embeddings).astype(np.float32)
    if len(meta) != len(emb):
        raise SystemExit(
            f"Metadata has {len(meta)} rows but embeddings have {len(emb)}. "
            "Run the preparation and embedding steps again."
        )
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)  # cosine = dot product now
    return meta, emb


def binned_vectors(spectra, bin_width=BIN_WIDTH, max_mz=MAX_FRAGMENT_MZ) -> np.ndarray:
    """Binned spectrum baseline: one vector per spectrum with the square root of
    the relative intensity summed per m/z bin, scaled to length 1 (so the dot
    product of two vectors is their cosine). No learning involved."""
    n_bins = int(np.ceil(max_mz / bin_width)) + 1
    vectors = np.zeros((len(spectra), n_bins), dtype=np.float32)
    for i, s in enumerate(spectra):
        bins = np.clip(np.floor(s["mz"] / bin_width).astype(int), 0, n_bins - 1)
        np.add.at(vectors[i], bins, np.sqrt(s["intensity"] / s["intensity"].max()))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def family_labels(meta: pd.DataFrame) -> np.ndarray:
    """GNPS family label per spectrum. Spectra alone in their family
    (singletons) get a unique negative label, so they never 'match' anything."""
    labels = meta["gnps_family"].to_numpy().copy()
    alone = meta["family_size"].to_numpy() < 2
    labels[alone] = -(np.arange(len(meta))[alone] + 1)
    return labels


def save_summary(rows, path) -> pd.DataFrame:
    """Save a (metric, value, note) table. Values are stored as text, because
    one column mixes numbers, percentages and settings."""
    def tidy(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3g}" if abs(value) < 1000 else f"{float(value):,.0f}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        return str(value)
    df = pd.DataFrame(rows, columns=["metric", "value", "note"])
    df["value"] = df["value"].map(tidy)
    df["note"] = df["note"].astype(str)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def write_json(data: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
