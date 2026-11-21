"""accuracy and time on the sen1floods11 hand labelled test chips.

for each chip we hide part of the image and compare four ways to run:

  clean      the real pixels everywhere (what you'd get with no clouds)
  zero_fill  hidden pixels set to the band mean, then the normal model
  drop_mean  hidden pixels blanked as in zero_fill, and patches that are
             mostly hidden never enter the encoder, filled with the mean token
  drop_zero  same, filled with zeros
  drop_near  same, filled with a copy of the closest kept patch
  dec_mean   check only: full encoder on the zero_fill image, then the hidden
             patches get the mean fill before the decoder

hidden parts come from two sources:
  synth  smooth random blobs covering a set fraction of the chip
  real   s2cloudless cloud mask on the chip itself (prob >= 0.4, dilated),
         made first with bench/cloud_masks.py

scores are on labelled pixels that are NOT hidden, because nobody can map
water under a cloud from optical data anyway.

  python bench/eval_sen1floods11.py --device mps --out runs/eval.json
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geofm_infer.models import FLOOD_S2_INDEX, load_flood_model, normalize, segment  # noqa: E402
from geofm_infer.tokens import patch_keep  # noqa: E402

DATA = os.path.expanduser("~/geofm-infer-data/sen1floods11")
PATCH = 16


def chip_ids(split):
    rows = csv.reader(open(os.path.join(DATA, f"flood_{split}_data.csv")))
    return [r[0].replace("_S1Hand.tif", "") for r in rows if r]


def read_chip(cid):
    with rasterio.open(f"{DATA}/S2Hand/{cid}_S2Hand.tif") as f:
        s2 = f.read().astype(np.float32) / 10000
    with rasterio.open(f"{DATA}/LabelHand/{cid}_LabelHand.tif") as f:
        y = f.read(1)
    return s2, y


def blob_mask(h, w, frac, seed):
    """True = hidden. smooth noise cut at the right quantile."""
    if frac <= 0:
        return np.zeros((h, w), bool)
    g = torch.Generator().manual_seed(seed)
    n = torch.rand(1, 1, h // 32 + 2, w // 32 + 2, generator=g)
    n = F.interpolate(n, size=(h + 64, w + 64), mode="bicubic", align_corners=False)
    n = n[0, 0, 32:32 + h, 32:32 + w].numpy()
    return n >= np.quantile(n, 1 - frac)


def cloud_mask(cid):
    """made by bench/cloud_masks.py (s2cloudless can't share a process with torch)."""
    return np.load(os.path.join(DATA, "clouds", f"{cid}.npy"))


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev.startswith("cuda"):
        torch.cuda.synchronize()


def run(model, x, hidden, mode, dev):
    """returns (pred, seconds)."""
    xt = normalize(torch.from_numpy(x)[None].to(dev))
    keep = None
    if mode != "clean":
        # hidden pixels are blanked in every mode, also inside patches that are
        # kept because they are mostly clear. a real cloud hides them too.
        xt = xt.masked_fill(torch.from_numpy(hidden).to(dev)[None, None], 0.0)
    if mode.startswith("drop") or mode == "dec_mean":
        clear = torch.from_numpy(~hidden).to(dev)[None]
        keep = patch_keep(clear, PATCH, min_clear=0.5)
    sync(dev)
    t = time.perf_counter()
    if keep is not None and not keep.any():
        out = None  # nothing visible, nothing to run
    else:
        fill = {"drop_zero": "zero", "drop_near": "nearest"}.get(mode, "mean")
        out = segment(model, xt, keep=keep, fill=fill, dense_encoder=(mode == "dec_mean"))
    sync(dev)
    dt = time.perf_counter() - t
    pred = None if out is None else out.argmax(1)[0].cpu().numpy()
    return pred, dt, (keep.float().mean().item() if keep is not None else 1.0)


def add(score, pred, y, scored):
    v = scored & (y >= 0)
    if pred is None:
        p = np.zeros_like(y)
    else:
        p = pred
    score["tp"] += int(((p == 1) & (y == 1) & v).sum())
    score["fp"] += int(((p == 1) & (y == 0) & v).sum())
    score["fn"] += int(((p == 0) & (y == 1) & v).sum())
    score["right"] += int(((p == y) & v).sum())
    score["n"] += int(v.sum())


def summary(s):
    iou = s["tp"] / max(s["tp"] + s["fp"] + s["fn"], 1)
    return {"water_iou": round(iou, 4), "acc": round(s["right"] / max(s["n"], 1), 4),
            "pixels": s["n"], "sec_per_chip": round(float(np.mean(s["t"])), 4),
            "kept_tokens": round(float(np.mean(s["kept"])), 3), "agree_clean": round(s["agree"] / max(s["n"], 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--modes", default="clean,zero_fill,drop_mean,drop_zero,drop_near")
    ap.add_argument("--out", default="runs/eval.json")
    a = ap.parse_args()

    model = load_flood_model(a.device)
    ids = chip_ids(a.split)
    if a.limit:
        ids = ids[:a.limit]
    modes = a.modes.split(",")
    settings = [("synth", float(f)) for f in a.fracs.split(",")] + [("real", None)]
    res = {f"{src}_{frac}": {m: {"tp": 0, "fp": 0, "fn": 0, "right": 0, "n": 0, "agree": 0, "t": [], "kept": []}
                             for m in modes} for src, frac in settings}

    # warm up so the first chip's time isn't all kernel compiling
    s2, _ = read_chip(ids[0])
    for m in modes:
        run(model, s2[FLOOD_S2_INDEX], np.zeros(s2.shape[1:], bool), m, a.device)

    cloudy_frac = []
    for i, cid in enumerate(ids):
        s2, y = read_chip(cid)
        x = s2[FLOOD_S2_INDEX]
        clean_pred, _, _ = run(model, x, np.zeros(y.shape, bool), "clean", a.device)
        for src, frac in settings:
            hidden = blob_mask(*y.shape, frac, seed=i) if src == "synth" else cloud_mask(cid)
            if src == "real":
                cloudy_frac.append(float(hidden.mean()))
            key = f"{src}_{frac}"
            for m in modes:
                if m == "clean":
                    pred, dt, kept = clean_pred, 0.0, 1.0
                    _, dt, _ = run(model, x, hidden, m, a.device)
                else:
                    pred, dt, kept = run(model, x, hidden, m, a.device)
                s = res[key][m]
                add(s, pred, y, ~hidden)
                s["t"].append(dt)
                s["kept"].append(kept)
                v = (~hidden) & (y >= 0)
                p = pred if pred is not None else np.zeros_like(y)
                s["agree"] += int(((p == clean_pred) & v).sum())
        print(f"{i + 1}/{len(ids)} {cid}", flush=True)

    out = {"device": a.device, "split": a.split, "chips": len(ids),
           "real_cloud_frac_mean": round(float(np.mean(cloudy_frac)), 4),
           "real_cloudy_chips": int(sum(c > 0.05 for c in cloudy_frac)),
           "results": {k: {m: summary(s) for m, s in v.items()} for k, v in res.items()}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in out["results"].items():
        print(k)
        for m, s in v.items():
            print(f"  {m:10s} iou {s['water_iou']:.4f} acc {s['acc']:.4f} agree {s['agree_clean']:.4f} "
                  f"kept {s['kept_tokens']:.2f} {s['sec_per_chip'] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
