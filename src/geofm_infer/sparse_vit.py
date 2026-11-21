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
    tokens of that tile and layer) or "zero".
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

    cls = (vit.cls_token + cls_pos).expand(b, -1, -1)
    h = torch.cat([cls, h], 1)

    outs = []
    for i, blk in enumerate(vit.blocks):
        h = blk(h, attn_mask=attn)
        if i == len(vit.blocks) - 1:
            h = vit.norm(h)
        outs.append(scatter_back(h, take, valid, n, fill))
    return outs


def scatter_back(h, take, valid, n, fill):
    """put packed tokens back on the full grid of n patches."""
    b, _, d = h.shape
    cls, tok = h[:, :1], h[:, 1:]
    vmask = valid[..., None].to(tok.dtype)
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
