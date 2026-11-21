"""s2cloudless cloud masks for the sen1floods11 chips, saved as .npy.

runs on its own because lightgbm (inside s2cloudless) and torch crash when
they load in the same process on macos (both bring their own openmp).

  python bench/cloud_masks.py test
"""
import csv
import os
import sys

import numpy as np
import rasterio
from s2cloudless import S2PixelCloudDetector

DATA = os.path.expanduser(os.environ.get("GEOFM_DATA", "~/geofm-infer-data/sen1floods11"))


def main(split):
    out = os.path.join(DATA, "clouds")
    os.makedirs(out, exist_ok=True)
    det = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)
    ids = [r[0].replace("_S1Hand.tif", "") for r in csv.reader(open(f"{DATA}/flood_{split}_data.csv")) if r]
    for cid in ids:
        with rasterio.open(f"{DATA}/S2Hand/{cid}_S2Hand.tif") as f:
            s2 = f.read().astype(np.float32) / 10000
        # B01 B02 B04 B05 B08 B8A B09 B10 B11 B12
        bands = s2[[0, 1, 3, 4, 7, 8, 9, 10, 11, 12]].transpose(1, 2, 0)[None]
        m = det.get_cloud_masks(bands)[0].astype(bool)
        np.save(os.path.join(out, f"{cid}.npy"), m)
        print(f"{cid} {m.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
