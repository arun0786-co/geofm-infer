"""time a whole sentinel-2 scene three ways and compare the maps.

  terratorch  terratorch's tiled_inference, the usual way to run these models
  dense       our tiler, every tile, every patch
  skip        our tiler, tiles with no clear patch skipped, the rest run whole
  drop        our tiler, those tiles skipped and cloudy patches dropped

agreement is measured against `dense` on clear pixels.

  python bench/scene_bench.py ~/geofm-infer-data/scenes/T43PFL_20180825 --device mps
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geofm_infer.models import load_flood_model, normalize, segment  # noqa: E402
from geofm_infer.scene import Plan, run_scene  # noqa: E402


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--crop", type=int, default=0, help="use only the top left crop x crop pixels")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-terratorch", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.expanduser(a.scene)
    dev = a.device

    stack = np.load(os.path.join(d, "stack.npy"))
    clear = np.load(os.path.join(d, "clear.npy"))
    if a.crop:
        stack, clear = stack[:, :a.crop, :a.crop], clear[:a.crop, :a.crop]
    x = normalize(torch.from_numpy(stack.astype(np.float32))[None])[0].numpy()
    # nodata outside the swath: zero after normalizing (the band mean)
    x[:, stack[0] == 0] = 0

    model = load_flood_model(dev)
    dtype = torch.float16 if a.fp16 else torch.float32
    if a.fp16:
        model = model.half()

    def fn(xb, kb):
        return segment(model, xb.to(dtype), keep=kb)

    plan = Plan(tile=a.tile, overlap=a.overlap, batch=a.batch)
    # warm up
    w = min(a.tile, clear.shape[0])
    run_scene(fn, x[:, :w, :w], clear[:w, :w], plan, device=dev)
    sync(dev)

    res = {"scene": os.path.basename(d.rstrip("/")), "device": dev, "shape": list(clear.shape),
           "cloud_frac": round(1 - float(clear.mean()), 4), "plan": vars(plan), "fp16": a.fp16}
    preds = {}
    for name, drop, skip in [("dense", False, False), ("skip", False, True), ("drop", True, True)]:
        sync(dev)
        t = time.perf_counter()
        pred, st = run_scene(fn, x, clear, plan, device=dev, drop=drop, skip=skip)
        sync(dev)
        res[name] = {"sec": round(time.perf_counter() - t, 2), **st}
        preds[name] = pred
        print(name, res[name], flush=True)

    if not a.skip_terratorch:
        from terratorch.tasks.tiled_inference import tiled_inference
        stride = a.tile - a.overlap
        xt = torch.from_numpy(x)[None]
        sync(dev)
        t = time.perf_counter()
        with torch.no_grad():
            out = tiled_inference(lambda b: segment(model, b.to(dtype)), xt, out_channels=2,
                                  crop=a.tile, stride=stride, batch_size=a.batch, device=dev)
        sync(dev)
        res["terratorch"] = {"sec": round(time.perf_counter() - t, 2)}
        preds["terratorch"] = out[0].argmax(0).cpu().numpy().astype(np.int16)
        print("terratorch", res["terratorch"], flush=True)

    ok = clear & (preds["dense"] >= 0)
    for name in preds:
        if name == "dense":
            continue
        p = preds[name]
        both = ok & (p >= 0)
        res[name]["agree_dense_clear"] = round(float((p[both] == preds["dense"][both]).mean()), 5)
        res[name]["coverage_clear"] = round(float(both.sum() / max(ok.sum(), 1)), 5)
    for name in preds:
        p = preds[name]
        res[name]["water_frac_clear"] = round(float((p[ok] == 1).mean()), 5)

    print(json.dumps(res, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        np.save(a.out.replace(".json", "_drop.npy"), preds["drop"])


if __name__ == "__main__":
    main()
