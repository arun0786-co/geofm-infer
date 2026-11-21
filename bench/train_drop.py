"""teach the decoder about dropped patches.

the encoder stays frozen (it is the pretrained prithvi, and it already copes
with missing patches). the neck, decoder and head of the sen1floods11 model
are trained a little more on the training chips, where each chip gets a
random hidden area (0 to 80%, smooth blobs like in the eval). hidden pixels
are blanked and patches that are mostly hidden are dropped with the mean
fill, exactly as at inference. 0% hidden is in the mix, so the dense model
should not get worse; the eval checks that.

  python bench/train_drop.py --device cuda --epochs 10 --out runs/drop_decoder.pt
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from eval_sen1floods11 import DATA, PATCH, blob_mask, chip_ids, read_chip  # noqa: E402
from geofm_infer.models import FLOOD_S2_INDEX, load_flood_model, normalize  # noqa: E402
from geofm_infer.sparse_vit import sparse_features  # noqa: E402
from geofm_infer.tokens import patch_keep  # noqa: E402


def load_split(split, limit=0):
    xs, ys = [], []
    ids = chip_ids(split)
    for cid in ids[:limit] if limit else ids:
        s2, y = read_chip(cid)
        xs.append(s2[FLOOD_S2_INDEX])
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def trainable(model):
    return [p for n, p in model.named_parameters() if not n.startswith("encoder.")]


def forward(model, x, hidden, max_frac):
    """x normalized (b, c, h, w), hidden (b, h, w) bool."""
    x = x.masked_fill(hidden[:, None], 0.0)
    keep = patch_keep(~hidden, PATCH, min_clear=0.5)
    keep[keep.sum(1) == 0, 0] = True  # a tile needs at least one patch
    with torch.no_grad():
        feats = sparse_features(model.encoder, x, keep, fill="mean")
    feats = model.neck(feats, image_size=x.shape[-2:])
    out = model.head(model.decoder(list(feats)))
    return F.interpolate(out, size=x.shape[-2:], mode="bilinear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/drop_decoder.pt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="first n chips only (smoke tests)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = a.device

    model = load_flood_model(dev)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    params = trainable(model)
    print(f"training {sum(p.numel() for p in params) / 1e6:.1f} M params, encoder frozen", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    xs, ys = load_split(a.split, a.limit)
    print(f"{len(xs)} {a.split} chips", flush=True)
    n = len(xs)
    step = 0
    for ep in range(a.epochs):
        model.neck.train(); model.decoder.train(); model.head.train()
        # batches are small (4), so keep the batch norm statistics from the
        # original fine tune instead of updating them
        for mod in model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
        order = list(range(n))
        random.shuffle(order)
        t, tot = time.time(), 0.0
        for i in range(0, n - a.batch + 1, a.batch):
            idx = order[i:i + a.batch]
            x = normalize(torch.from_numpy(xs[idx]).to(dev))
            y = torch.from_numpy(ys[idx]).long().to(dev)
            # random flips, as in the original fine tune
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            fr = [random.uniform(0, a.max_frac) if random.random() < 0.8 else 0.0 for _ in idx]
            hidden = torch.from_numpy(np.stack([blob_mask(512, 512, f, seed=10_000 * ep + j + 7)
                                                for j, f in zip(idx, fr)])).to(dev)
            logits = forward(model, x, hidden, a.max_frac)
            target = y.masked_fill(hidden, -1)  # nothing to learn under a cloud
            loss = F.cross_entropy(logits, target, ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            step += 1
        print(f"epoch {ep} loss {tot / max(n // a.batch, 1):.4f} {time.time() - t:.0f}s", flush=True)

    keep = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("encoder.")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state": keep, "args": vars(a)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
