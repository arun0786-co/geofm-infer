"""run a prithvi vit on the kept patches only.

prithvi was pretrained as a masked autoencoder, so its encoder already knows
how to work with part of the image missing. here we use that at inference:
cloudy and nodata patches never enter the transformer. the outputs are put
back on the full grid afterwards, and the dropped places get a fill value,
so the decoder (upernet etc.) runs unchanged.
"""
import torch

from .tokens import pack_order


def embed(vit, x):
    """patch embedding + position embedding for all patches. (b, n, d)"""
    if x.dim() == 4 and vit.patch_embed.input_size[0] == 1:
        x = x.unsqueeze(2)
    shape = x.shape[-3:]
    t = vit.patch_embed(x)
    pos = vit.interpolate_pos_encoding(shape)
    return t + pos[:, 1:, :], pos[:, :1, :]


def sparse_features(vit, x, keep, fill="mean"):
    """like vit.forward_features(x), but only kept patches are computed.

    keep: (b, n) bool. returns a list with one (b, 1 + n, d) tensor per block,
    the same layout forward_features gives. fill is "mean" (mean of the kept
    tokens of that tile and layer), "zero", or "nearest" (copy of the closest
    kept patch).
    """
    tok, cls_pos = embed(vit, x)
    b, n, d = tok.shape
    idx, lengths, m = pack_order(keep)
    take = idx[:, :m]
    h = torch.gather(tok, 1, take[..., None].expand(-1, -1, d))

    # True means "attend to this key". the cls token is always there.
    valid = torch.arange(m, device=x.device)[None, :] < lengths[:, None]
    attn = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), valid], 1)
    attn = attn[:, None, None, :]
    if bool(valid.all()):
        attn = None  # no padding, so the fast attention path can be used
    near = nearest_kept(keep, grid_shape(vit, x)) if fill == "nearest" else None

    cls = (vit.cls_token + cls_pos).expand(b, -1, -1)
    h = torch.cat([cls, h], 1)

    outs = []
    for i, blk in enumerate(vit.blocks):
        h = blk(h, attn_mask=attn)
        if i == len(vit.blocks) - 1:
            h = vit.norm(h)
        outs.append(scatter_back(h, take, valid, n, fill, near))
    return outs


def grid_shape(vit, x):
    p = vit.patch_embed.patch_size
    return x.shape[-2] // p[-2], x.shape[-1] // p[-1]


def nearest_kept(keep, grid):
    """for every patch, the index of the closest kept patch. (b, n)"""
    gh, gw = grid
    yy, xx = torch.meshgrid(torch.arange(gh, device=keep.device),
                            torch.arange(gw, device=keep.device), indexing="ij")
    pos = torch.stack([yy.flatten(), xx.flatten()], 1).float()
    dist = torch.cdist(pos, pos)                       # (n, n)
    dist = dist[None].expand(keep.shape[0], -1, -1).clone()
    dist.masked_fill_(~keep[:, None, :], float("inf"))
    return dist.argmin(-1)


def scatter_back(h, take, valid, n, fill, near=None):
    """put packed tokens back on the full grid of n patches."""
    b, _, d = h.shape
    cls, tok = h[:, :1], h[:, 1:]
    vmask = valid[..., None].to(tok.dtype)
    if fill == "nearest":
        full = scatter_back(h, take, valid, n, "zero")[:, 1:]
        full = torch.gather(full, 1, near[..., None].expand(-1, -1, d))
        return torch.cat([cls, full], 1)
    if fill == "mean":
        cnt = vmask.sum(1).clamp(min=1)
        base = (tok * vmask).sum(1, keepdim=True) / cnt[:, None]
    elif fill == "zero":
        base = tok.new_zeros(b, 1, d)
    else:
        raise ValueError(f"unknown fill {fill}")
    full = base.expand(b, n, d).clone()
    # padding slots write the fill value onto dropped places, which is a no-op
    src = tok * vmask + base * (1 - vmask)
    full.scatter_(1, take[..., None].expand(-1, -1, d), src)
    return torch.cat([cls, full], 1)


def fill_dropped(feats, keep, fill="mean"):
    """take normal dense features and overwrite the dropped places with the fill.

    not a speed up. it is a check: the decoder then sees the same fill as in
    sparse_features, but the kept tokens had full context in the encoder.
    """
    b, n = keep.shape
    idx, lengths, _ = pack_order(keep, multiple=1)
    take = idx[:, :n]
    valid = torch.arange(n, device=keep.device)[None, :] < lengths[:, None]
    out = []
    for h in feats:
        packed = torch.cat([h[:, :1], torch.gather(h[:, 1:], 1, take[..., None].expand(-1, -1, h.shape[-1]))], 1)
        out.append(scatter_back(packed, take, valid, n, fill))
    return out
