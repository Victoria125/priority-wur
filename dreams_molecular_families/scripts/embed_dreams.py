"""Step 2. Compute DreaMS embeddings for the cleaned spectra.

Run this script with the Python of the DreaMS environment (Python 3.11).
The first run downloads the DreaMS model weights from Hugging Face.

If the embeddings already exist for exactly the same cleaned MGF file,
nothing is computed again (use --force to recompute).
"""
import argparse
import platform
import time
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

import common as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    p = C.run_paths(args.run)
    mgf_md5 = C.md5sum(p.clean_mgf)
    if not args.force and p.embeddings.exists() and p.embeddings_info.exists():
        if C.read_json(p.embeddings_info).get("clean_mgf_md5") == mgf_md5:
            print(f"{p.embeddings.name} is up to date for {args.run}. Nothing to do.")
            return

    import torch
    from dreams.api import dreams_embeddings

    if torch.cuda.is_available():
        device = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "CPU (this can take a while; a GPU is much faster)"
    meta = pd.read_parquet(p.metadata)
    print(f"Computing DreaMS embeddings for {len(meta):,} spectra on {device}...")

    start = time.time()
    # DreaMS reads the MGF file and returns one 1024-dimensional vector per
    # spectrum, in the same order as the file.
    embeddings = np.asarray(dreams_embeddings(str(p.clean_mgf), batch_size=args.batch_size), dtype=np.float32)
    seconds = time.time() - start

    if embeddings.shape[0] != len(meta):
        raise SystemExit(f"Got {embeddings.shape[0]} embeddings for {len(meta)} spectra. Something is wrong.")
    if not np.isfinite(embeddings).all():
        raise SystemExit("Some embeddings contain NaN or infinite values.")

    np.save(p.embeddings, embeddings)
    C.write_json({
        "clean_mgf_md5": mgf_md5,
        "shape": list(embeddings.shape),
        "model": "DreaMS_embedding (pre-trained, no fine-tuning)",
        "dreams_commit": C.DREAMS_COMMIT,
        "device": device.split(" (")[0],
        "seconds": round(seconds),
        "computed_on": datetime.now().isoformat(timespec="seconds"),
        "machine": platform.platform(),
    }, p.embeddings_info)
    print(f"Done in {seconds:.0f} s. Shape {embeddings.shape} saved to {p.embeddings}")

    # Save the exact package versions used for the embeddings (reproducibility).
    versions = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    (C.EXPERIMENT_DIR / "env_versions.txt").write_text("\n".join(versions) + "\n")


if __name__ == "__main__":
    main()
