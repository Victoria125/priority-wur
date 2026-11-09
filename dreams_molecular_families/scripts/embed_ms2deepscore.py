"""Step 2b. MS2DeepScore embeddings for the same spectra (a baseline for A and B).

MS2DeepScore (van der Hooft lab, WUR) is trained to predict the structural
similarity of two molecules from their spectra. We use the pre-trained model
its authors recommend (Zenodo record 17826815: trained on > 500,000 spectra
from GNPS, MoNA, MassBank and MSnLib; works in both ion modes).

Run this script with the Python of the MS2DeepScore environment (.ms2ds_env).
It is separate from the DreaMS environment because the two packages pin
different versions of numpy, torch and matchms.

Every method gets the same input: the 100 most intense peaks that DreaMS's
embedding model keeps (common.top_peaks). If the embeddings already exist for
exactly the same spectra, nothing is computed again (use --force to recompute).
"""
import argparse
import time
from datetime import datetime
from importlib import metadata

import numpy as np
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

import common as C

MODEL_URL = "https://zenodo.org/records/17826815/files/ms2deepscore_model.pt?download=1"
MODEL_PATH = C.EXPERIMENT_DIR / ".ms2deepscore" / "ms2deepscore_model.pt"


def to_matchms(spec) -> Spectrum:
    """Our spectrum dict -> matchms Spectrum, with the same 100 peaks DreaMS sees.
    All our spectra are positive mode; the charge is the one GNPS gives (0 = unknown),
    which MS2DeepScore does not use."""
    spec = C.top_peaks(spec)
    order = np.argsort(spec["mz"])
    metadata = {"precursor_mz": float(spec["precursor_mz"]), "ionmode": "positive"}
    if spec.get("charge"):
        metadata["charge"] = int(spec["charge"])
    s = Spectrum(
        mz=spec["mz"][order].astype(float),
        intensities=spec["intensity"][order].astype(float),
        metadata=metadata,
    )
    return normalize_intensities(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--variant", default="protocol", choices=list(C.VARIANTS))
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    p = C.run_paths(args.run, args.variant)
    if not p.clean_mgf.exists():
        raise SystemExit(f"{args.run} has no spectra for the '{args.variant}' variant. Nothing to embed.")
    fingerprint = C.spectra_fingerprint(p.clean_mgf)
    info = C.read_json(p.ms2ds_info) if p.ms2ds_info.exists() else {}
    if (not args.force and p.ms2ds_embeddings.exists() and info.get("spectra_fingerprint") == fingerprint
            and info.get("input_peaks") == C.INPUT_PEAKS):
        print(f"{p.ms2ds_embeddings.name} is up to date for {args.run}. Nothing to do.")
        return

    if not MODEL_PATH.exists():
        print("Downloading the MS2DeepScore model (416 MB) from Zenodo...")
        C.download(MODEL_URL, MODEL_PATH)
    model = load_model(str(MODEL_PATH))

    spectra = [to_matchms(s) for s in C.read_mgf(p.clean_mgf)]
    print(f"Computing MS2DeepScore embeddings for {len(spectra):,} spectra...")
    start = time.time()
    embeddings = np.asarray(MS2DeepScore(model).get_embedding_array(spectra), dtype=np.float32)
    seconds = time.time() - start

    if embeddings.shape[0] != len(spectra):
        raise SystemExit(f"Got {embeddings.shape[0]} embeddings for {len(spectra)} spectra. Something is wrong.")
    if not np.isfinite(embeddings).all():
        raise SystemExit("Some embeddings contain NaN or infinite values.")

    np.save(p.ms2ds_embeddings, embeddings)
    C.write_json({
        "clean_mgf_md5": C.md5sum(p.clean_mgf),
        "spectra_fingerprint": fingerprint,
        "input_peaks": C.INPUT_PEAKS,
        "shape": list(embeddings.shape),
        "model": "MS2DeepScore pre-trained (Zenodo 17826815)",
        "ms2deepscore_version": metadata.version("ms2deepscore"),
        "seconds": round(seconds),
        "computed_on": datetime.now().isoformat(timespec="seconds"),
    }, p.ms2ds_info)
    print(f"Done in {seconds:.0f} s. Shape {embeddings.shape} saved to {p.ms2ds_embeddings}")


if __name__ == "__main__":
    main()
