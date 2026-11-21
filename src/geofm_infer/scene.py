"""whole scene inference: tiles, batching and stitching.

the scene is cut into overlapping tiles. tiles with no clear patch are
skipped. the rest are sorted by how many patches they keep, so tiles in one
batch have about the same length and little padding is wasted. logits from
overlapping tiles are blended with a weight that is high in the middle of a
tile and low at its edges, which hides most seams.
"""
from dataclasses import dataclass

import numpy as np
import torch

from .tokens import patch_keep


@dataclass
class Plan:
    tile: int = 512
    overlap: int = 64
    patch: int = 16
    min_clear: float = 0.5
    batch: int = 8


def tile_starts(size, tile, overlap):
    """starts along one axis so tiles cover [0, size) and the last one ends at size."""
    if size <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, size - tile, step))
    starts.append(size - tile)
    return starts


def blend_weight(tile, overlap):
    """(tile, tile) weight: 1 in the middle, ramps down to a small value at the edges."""
    if overlap <= 0:
        return np.ones((tile, tile), np.float32)
    ramp = np.ones(tile, np.float32)
    r = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    ramp[:overlap] = r
    ramp[-overlap:] = r[::-1]
    return np.outer(ramp, ramp)


def run_scene(model_fn, image, clear, plan=Plan(), classes=2, device="cpu", drop=True, skip=True):
    """image: (c, h, w) float32 already normalized. clear: (h, w) bool.

    model_fn(x, keep) -> logits (b, classes, tile, tile). keep is None when
    drop is False. skip=True leaves out tiles with no clear patch; with
    skip=False every tile runs (the plain dense way). returns (pred (h, w) int16 with -1 where nothing was run,
    stats dict).
    """
    if drop and not skip:
        raise ValueError("drop needs skip: a tile with nothing kept can't run sparse")
    c, h, w = image.shape
    t = plan.tile
    pad_h, pad_w = max(t - h, 0), max(t - w, 0)
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)))
        clear = np.pad(clear, ((0, pad_h), (0, pad_w)))
    H, W = clear.shape

    tiles = [(y, x) for y in tile_starts(H, t, plan.overlap) for x in tile_starts(W, t, plan.overlap)]
    keeps = {}
    for y, x in tiles:
        cl = torch.from_numpy(clear[y:y + t, x:x + t])[None]
        k = patch_keep(cl, plan.patch, plan.min_clear)[0]
        if k.any() or not skip:
            keeps[(y, x)] = k
    todo = sorted(keeps, key=lambda yx: int(keeps[yx].sum()))

    acc = np.zeros((classes, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    wt = blend_weight(t, plan.overlap)
    for i in range(0, len(todo), plan.batch):
        group = todo[i:i + plan.batch]
        xb = torch.from_numpy(np.stack([image[:, y:y + t, x:x + t] for y, x in group])).to(device)
        kb = None
        if drop:
            kb = torch.stack([keeps[g] for g in group]).to(device)
            if bool(kb.all()):
                kb = None  # nothing to drop, the dense path is faster
        out = model_fn(xb, kb).float().softmax(1).cpu().numpy()
        for (y, x), o in zip(group, out):
            acc[:, y:y + t, x:x + t] += o * wt
            wsum[y:y + t, x:x + t] += wt

    pred = np.where(wsum > 0, acc.argmax(0), -1).astype(np.int16)[:h, :w]
    total = len(tiles) * (t // plan.patch) ** 2
    ran = sum(int(k.sum()) for k in keeps.values()) if drop else len(keeps) * (t // plan.patch) ** 2
    stats = {"tiles": len(tiles), "tiles_run": len(keeps), "tokens_total": total, "tokens_run": ran}
    return pred, stats
