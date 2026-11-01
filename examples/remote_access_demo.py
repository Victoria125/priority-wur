"""Smoke test / demo: use datasets without downloading them.

Run from the project root:
    .venv\\Scripts\\python.exe examples\\remote_access_demo.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from priority_data import remote  # noqa: E402

ACC = "MSV000094769"


def main():
    t0 = time.time()
    files = remote.massive_files(ACC, collection="ccms_peak")
    genomes = remote.massive_files(ACC, suffix=".fasta")
    print(f"{ACC}: {len(files)} mzML in ccms_peak, {len(genomes)} genome FASTA files "
          f"(listing took {time.time() - t0:.1f}s)")

    smallest = min(files, key=lambda f: f["size"] or 1e18)
    t0 = time.time()
    ms2 = list(remote.iter_spectra(ACC, smallest["path"], ms_level=2))
    print(f"Streamed {smallest['path']} ({smallest['size'] / 1e6:.0f} MB) in {time.time() - t0:.0f}s: "
          f"{len(ms2)} MS/MS spectra, raw file not kept")
    if ms2:
        s = ms2[0]
        print(f"  first MS/MS: scan {s['scan']}, precursor m/z {s['precursor_mz']:.4f}, "
              f"{len(s['mz'])} peaks")
        usi = f"mzspec:{ACC}:{Path(smallest['path']).stem}:scan:{s['scan']}"
        try:
            hit = remote.usi_spectrum(usi)
            print(f"  same spectrum via USI service: {len(hit['peaks'])} peaks ({usi})")
        except Exception as exc:
            print(f"  USI lookup failed for {usi}: {exc}")

    t0 = time.time()
    fasta = remote.genome_fasta(Path(genomes[0]["path"]).stem)
    print(f"Genome {Path(genomes[0]['path']).stem} -> {fasta.relative_to(remote.ROOT)} "
          f"({time.time() - t0:.0f}s)")

    runs = remote.rnaseq_runs("PRJNA1397920")  # S. xinjiangensis XJ54 time course
    total = sum(r["fastq_bytes"] for r in runs)
    t0 = time.time()
    reads = list(remote.iter_fastq(runs[0]["fastq_urls"][0], limit=1000))
    print(f"XJ54 RNA-seq: {len(runs)} runs, {total / 1e9:.1f} GB of FASTQ; streamed "
          f"{len(reads)} reads of {runs[0]['run_accession']} ({runs[0]['sample_title']}) "
          f"in {time.time() - t0:.0f}s, first read length {len(reads[0][1])}")


if __name__ == "__main__":
    main()
