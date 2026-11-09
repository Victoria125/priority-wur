"""Embeddings from a self-supervised DreaMS model (no fine-tuning).

For a fair comparison with our model trained from scratch, both models are
read the same way: the hidden state of the precursor token in the last
transformer layer, for the protocol spectra of a run (DreaMS's own
dreams_intermediates function).

  --name dreams_ssl         the published DreaMS self-supervised model (ssl_model.ckpt)
  --name scratch_small      our model, with --ckpt pointing to its final.ckpt

The vectors are saved as runs/<run>/data/<name>_embeddings_protocol.npy, with
a .json that records the MGF md5 and the checkpoint.

Usage:
  python scripts/embed_ssl.py --run tahlan_streptomyces --name dreams_ssl
  python scripts/embed_ssl.py --run tahlan_streptomyces --name scratch_small --ckpt runs_scratch/small/final.ckpt
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import common as C


def output_paths(p, name):
    return p.data / f"{name}_embeddings_protocol.npy", p.data / f"{name}_embeddings_protocol.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="name of the run (see common.RUNS)")
    parser.add_argument("--name", required=True, help="dreams_ssl, or a name for your own model")
    parser.add_argument("--ckpt", type=Path, default=None, help="model checkpoint (not needed for dreams_ssl)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    p = C.run_paths(args.run, "protocol")
    if not p.clean_mgf.exists():
        raise SystemExit(f"{args.run} has no protocol spectra. Nothing to embed.")
    emb_path, info_path = output_paths(p, args.name)
    fingerprint = C.spectra_fingerprint(p.clean_mgf)
    ckpt = args.ckpt
    if args.name == "dreams_ssl" and ckpt is None:
        from dreams import utils
        from dreams.definitions import PRETRAINED
        ckpt = PRETRAINED / "ssl_model.ckpt"
        if not ckpt.exists():
            utils.download_pretrained_model("ssl_model.ckpt")
    if ckpt is None or not Path(ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    if emb_path.exists() and info_path.exists():
        info = C.read_json(info_path)
        if info.get("spectra_fingerprint") == fingerprint and info.get("ckpt_md5") == C.md5sum(ckpt):
            print(f"{emb_path.name} is up to date for {args.run}. Nothing to do.")
            return

    from dreams.api import dreams_intermediates

    start = time.time()
    embeddings = np.asarray(dreams_intermediates(
        model=str(ckpt), msdata=str(p.clean_mgf), precursor_only=True, batch_size=args.batch_size,
        compute_embeddings=True, compute_attn_matrices=False, n_highest_peaks=60,
    ), dtype=np.float32)
    seconds = time.time() - start
    n_spectra = len(C.read_mgf(p.clean_mgf))
    if embeddings.shape[0] != n_spectra or not np.isfinite(embeddings).all():
        raise SystemExit(f"Bad embeddings: shape {embeddings.shape} for {n_spectra} spectra, or NaN values.")

    np.save(emb_path, embeddings)
    C.write_json({
        "spectra_fingerprint": fingerprint, "ckpt": str(ckpt), "ckpt_md5": C.md5sum(ckpt),
        "representation": "precursor token, last transformer layer (dreams_intermediates)",
        "shape": list(embeddings.shape), "seconds": round(seconds),
        "computed_on": datetime.now().isoformat(timespec="seconds"),
    }, info_path)
    print(f"Done in {seconds:.0f} s. Shape {embeddings.shape} saved to {emb_path}")


if __name__ == "__main__":
    main()
