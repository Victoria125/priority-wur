"""Step 0. Does our DreaMS installation reproduce a published DreaMS result?

Before we compare DreaMS with other methods on our data, we check that our
setup gives the same numbers as the DreaMS networking tutorial
(.DreaMS/tutorials/networking.ipynb, run by the DreaMS authors and saved with
its outputs). The tutorial uses one public file, S_N1.mzML from MassIVE
MSV000086206, and reports:
  - 3,809 MS/MS spectra with DreaMS embeddings
  - DreaMS similarity 0.9109248 between spectra 1137 and 1165
  - 6,858 edges in the DreaMS k-NN network (k = 3, similarity >= 0.7)

We follow the tutorial code exactly (MSData.load on the mzML file, then
dreams_embeddings), so no cleaning of ours is involved.
"""
import json
import time
from datetime import datetime

import numpy as np
from sklearn.neighbors import kneighbors_graph

import common as C

MZML_URL = (
    "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
    "file=f.MSV000086206/peak/mzml/S_N1.mzML&forceDownload=true"
)
EXPECTED = {"spectra": 3809, "pair": (1137, 1165), "pair_similarity": 0.9109248, "network_edges": 6858}
OUT_DIR = C.EXPERIMENT_DIR / "reproduction"


def main():
    from dreams.api import dreams_embeddings
    from dreams.utils.data import MSData

    data_dir = C.EXPERIMENT_DIR / ".DreaMS" / "data"
    mzml = data_dir / "S_N1.mzML"
    if not mzml.exists():
        print("Downloading S_N1.mzML (49 MB) from MassIVE...")
        C.download(MZML_URL, mzml)

    start = time.time()
    msdata = MSData.load(mzml)
    embs = np.asarray(dreams_embeddings(msdata), dtype=np.float64)
    seconds = time.time() - start

    # Same steps as the tutorial: cosine similarity and the k-NN network.
    unit = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    i, j = EXPECTED["pair"]
    pair_similarity = float(unit[i] @ unit[j])
    A = kneighbors_graph(embs, 3, mode="distance", metric="cosine", include_self=False).toarray()
    A = np.where(A != 0, 1 - A, 0)
    A[A < 0.7] = 0
    edges = int(np.count_nonzero(np.triu((A + A.T) > 0, k=1)))  # undirected, as networkx counts them

    got = {"spectra": int(embs.shape[0]), "pair_similarity": round(pair_similarity, 7), "network_edges": edges}
    checks = {
        "spectra": got["spectra"] == EXPECTED["spectra"],
        "pair_similarity": abs(pair_similarity - EXPECTED["pair_similarity"]) < 1e-4,
        "network_edges": got["network_edges"] == EXPECTED["network_edges"],
    }
    result = {
        "expected (DreaMS tutorial)": {k: v for k, v in EXPECTED.items() if k != "pair"},
        "pair": list(EXPECTED["pair"]),
        "got (this machine)": got,
        "match": checks,
        "reproduced": all(checks.values()),
        "seconds": round(seconds),
        "dreams_commit": C.DREAMS_COMMIT,
        "computed_on": datetime.now().isoformat(timespec="seconds"),
    }
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "dreams_networking_tutorial.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
