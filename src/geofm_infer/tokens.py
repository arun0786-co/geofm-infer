"""which patches to keep.

a patch is kept when enough of its pixels are clear (not cloud, not nodata).
everything here works on a pixel mask where True means "clear".
"""
import torch
import torch.nn.functional as F


def patch_keep(clear: torch.Tensor, patch: int, min_clear: float = 0.5) -> torch.Tensor:
    """clear: (b, h, w) bool. returns (b, h//patch * w//patch) bool, row major."""
    frac = F.avg_pool2d(clear.float().unsqueeze(1), patch, patch)
    return (frac.flatten(1) >= min_clear)


def pack_order(keep: torch.Tensor, multiple: int = 8):
    """indices that move kept patches to the front of each row.

    returns (idx, lengths, padded_len). idx[:, :padded_len] is what goes into
    the transformer; positions past lengths[i] are padding.
    """
    # stable sort keeps the kept patches in their original order
    idx = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    lengths = keep.sum(1)
    n = int(lengths.max().item()) if keep.numel() else 0
    padded = min(keep.shape[1], -(-max(n, 1) // multiple) * multiple)
    return idx, lengths, padded
