"""Step 2. Compute DreaMS embeddings for the cleaned spectra.

Run this script with the Python of the DreaMS environment (Python 3.11).
The first run downloads the DreaMS model weights from Hugging Face.

If the embeddings already exist for exactly the same spectra (same
spectra fingerprint, see common.spectra_fingerprint), nothing is computed
again (use --force to recompute).
"""
import argparse
import platform
import time
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

import common as C


def spectrum_key(spec):
    """Two spectra with the same key are identical in the MGF file (same cluster,
    precursor and peaks), so DreaMS gives them the same embedding."""
    return spec["scans"], spec["precursor_mz"], spec["mz"].tobytes(), spec["intensity"].tobytes()


def reusable_embeddings(run, variant, spectra):
    """Embeddings of identical spectra in another, up-to-date variant: {row: vector}."""
    q = C.run_paths(run, variant)
    if not (q.embeddings.exists() and q.embeddings_info.exists() and q.clean_mgf.exists()):
        return {}
    if not C.embeddings_up_to_date(q.embeddings_info, q.clean_mgf, expected={
        "model": "DreaMS_embedding (pre-trained, no fine-tuning)",
        "dreams_commit": C.DREAMS_COMMIT,
    }):
        return {}
    row_of = {spectrum_key(s): j for j, s in enumerate(C.read_mgf(q.clean_mgf))}
    other = np.load(q.embeddings)
    return {i: other[row_of[spectrum_key(s)]] for i, s in enumerate(spectra) if spectrum_key(s) in row_of}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    parser.add_argument("--variant", default="protocol", choices=list(C.VARIANTS))
    parser.add_argument("--reuse-from", default=None, choices=list(C.VARIANTS),
                        help="copy the embeddings of identical spectra from another variant and embed only the rest")
    args = parser.parse_args()

    p = C.run_paths(args.run, args.variant)
    if not p.clean_mgf.exists():
        raise SystemExit(f"{args.run} has no spectra for the '{args.variant}' variant. Nothing to embed.")
    fingerprint = C.spectra_fingerprint(p.clean_mgf)
    if not args.force and p.embeddings.exists() and C.embeddings_up_to_date(
        p.embeddings_info, p.clean_mgf, expected={
            "model": "DreaMS_embedding (pre-trained, no fine-tuning)",
            "dreams_commit": C.DREAMS_COMMIT,
        }
    ):
        print(f"{p.embeddings.name} is up to date for {args.run}. Nothing to do.")
        return

    import torch
    from dreams.api import dreams_embeddings

    if torch.cuda.is_available():
        device = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "CPU (this can take a while; a GPU is much faster)"
    meta = pd.read_parquet(p.metadata)
    spectra = C.read_mgf(p.clean_mgf)
    reused = reusable_embeddings(args.run, args.reuse_from, spectra) if args.reuse_from else {}
    todo = [i for i in range(len(spectra)) if i not in reused]
    print(f"Computing DreaMS embeddings for {len(todo):,} of {len(meta):,} spectra on {device} "
          f"({len(reused):,} identical spectra reused from '{args.reuse_from}')...")

    start = time.time()
    embeddings = np.zeros((len(spectra), 1024), dtype=np.float32)
    for i, vector in reused.items():
        embeddings[i] = vector
    if todo:
        # DreaMS reads an MGF file and returns one 1024-dimensional vector per
        # spectrum, in the same order as the file.
        source = p.clean_mgf
        if reused:
            source = p.data / "_spectra_to_embed.mgf"
            C.write_mgf([spectra[i] for i in todo], source)
        embeddings[todo] = np.asarray(dreams_embeddings(str(source), batch_size=args.batch_size), dtype=np.float32)
        if source != p.clean_mgf:
            source.unlink()
            source.with_suffix(".hdf5").unlink(missing_ok=True)
    seconds = time.time() - start

    if embeddings.shape[0] != len(meta):
        raise SystemExit(f"Got {embeddings.shape[0]} embeddings for {len(meta)} spectra. Something is wrong.")
    if not np.isfinite(embeddings).all():
        raise SystemExit("Some embeddings contain NaN or infinite values.")

    np.save(p.embeddings, embeddings)
    C.write_json({
        "clean_mgf_md5": C.md5sum(p.clean_mgf),
        "spectra_fingerprint": fingerprint,
        "shape": list(embeddings.shape),
        "model": "DreaMS_embedding (pre-trained, no fine-tuning)",
        "dreams_commit": C.DREAMS_COMMIT,
        "device": device.split(" (")[0],
        "seconds": round(seconds),
        "computed": len(todo),
        "reused from variant": args.reuse_from if reused else None,
        "reused (identical spectra)": len(reused),
        "computed_on": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.platform(),
    }, p.embeddings_info)
    print(f"Done in {seconds:.0f} s. Shape {embeddings.shape} saved to {p.embeddings}")

    # Save the exact package versions used for the embeddings (reproducibility).
    versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    (C.EXPERIMENT_DIR / "env_versions.txt").write_text("\n".join(versions) + "\n")


if __name__ == "__main__":
    main()
