"""turn a sentinel-2 L1C tile (13 jp2 bands) into what scene_bench.py reads.

writes, next to the bands:
  stack.npy   (6, h, w) float16 reflectance: B02 B03 B04 B8A B11 B12 at 10 m
  clear.npy   (h, w) bool, True where s2cloudless says clear
  meta.json   crs and transform of the 10 m grid

s2cloudless runs at 60 m (as its authors suggest) and the mask is scaled up.
this script must not import torch (see cloud_masks.py).

  python bench/scene_prep.py ~/geofm-infer-data/scenes/T43PFL_20180825
"""
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from s2cloudless import S2PixelCloudDetector

MODEL_BANDS = ["02", "03", "04", "8A", "11", "12"]
CLOUD_BANDS = ["01", "02", "04", "05", "08", "8A", "09", "10", "11", "12"]


def read(path, size):
    with rasterio.open(path) as f:
        a = f.read(1, out_shape=(size, size), resampling=Resampling.bilinear)
    return a.astype(np.float32) / 10000


def main(d):
    with rasterio.open(os.path.join(d, "B02.jp2")) as f:
        size, crs, tr = f.width, f.crs, f.transform
    small = size // 6
    cb = np.stack([read(os.path.join(d, f"B{b}.jp2"), small) for b in CLOUD_BANDS], -1)
    det = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)
    cloud = det.get_cloud_masks(cb[None])[0].astype(bool)
    # nodata (outside the swath) is also not clear
    cloud |= cb[..., 1] == 0
    cloud = np.repeat(np.repeat(cloud, 6, 0), 6, 1)
    cloud = np.pad(cloud, ((0, size - cloud.shape[0]), (0, size - cloud.shape[1])), mode="edge")
    np.save(os.path.join(d, "clear.npy"), ~cloud)

    stack = np.stack([read(os.path.join(d, f"B{b}.jp2"), size) for b in MODEL_BANDS]).astype(np.float16)
    np.save(os.path.join(d, "stack.npy"), stack)
    json.dump({"crs": str(crs), "transform": list(tr)[:6], "size": size}, open(os.path.join(d, "meta.json"), "w"))
    print(f"size {size}, cloud {cloud.mean():.3f}")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1]))
