import torch
from terratorch.models.backbones.prithvi_mae import PrithviViT

from geofm_infer.sparse_vit import sparse_features
from geofm_infer.tokens import patch_keep, pack_order


def tiny_vit():
    torch.manual_seed(0)
    vit = PrithviViT(img_size=64, patch_size=(1, 16, 16), in_chans=6,
                     embed_dim=32, depth=3, num_heads=4)
    return vit.eval()


def test_patch_keep():
    clear = torch.zeros(1, 32, 32, dtype=torch.bool)
    clear[:, :16, :16] = True        # first patch clear
    clear[:, 16:, 16:24] = True      # last patch half clear
    keep = patch_keep(clear, 16, min_clear=0.5)
    assert keep.tolist() == [[True, False, False, True]]


def test_pack_order_keeps_order():
    keep = torch.tensor([[False, True, False, True], [True, True, True, False]])
    idx, lengths, m = pack_order(keep, multiple=1)
    assert idx[0, :2].tolist() == [1, 3]
    assert idx[1, :3].tolist() == [0, 1, 2]
    assert lengths.tolist() == [2, 3] and m == 3


def test_all_kept_matches_dense():
    vit = tiny_vit()
    x = torch.randn(2, 6, 64, 64)
    keep = torch.ones(2, 16, dtype=torch.bool)
    with torch.no_grad():
        dense = vit.forward_features(x)
        sparse = sparse_features(vit, x, keep)
    for a, b in zip(dense, sparse):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_dropped_pixels_do_not_matter():
    vit = tiny_vit()
    x = torch.randn(1, 6, 64, 64)
    keep = torch.ones(1, 16, dtype=torch.bool)
    keep[0, [0, 5, 6, 15]] = False
    y = x.clone()
    y[..., :16, :16] = 100.0          # patch 0 is dropped, so this has no effect
    with torch.no_grad():
        a = sparse_features(vit, x, keep)[-1]
        b = sparse_features(vit, y, keep)[-1]
    torch.testing.assert_close(a, b)


def test_batch_padding_matches_single():
    # tiles with different numbers of kept patches share a padded batch;
    # each must come out the same as when it runs alone
    vit = tiny_vit()
    x = torch.randn(3, 6, 64, 64)
    keep = torch.rand(3, 16) > 0.4
    keep[:, 0] = True
    with torch.no_grad():
        batch = sparse_features(vit, x, keep)[-1]
        for i in range(3):
            one = sparse_features(vit, x[i:i + 1], keep[i:i + 1])[-1]
            torch.testing.assert_close(batch[i:i + 1], one, atol=1e-5, rtol=1e-5)


def test_nearest_fill_copies_neighbour():
    vit = tiny_vit()
    x = torch.randn(1, 6, 64, 64)
    keep = torch.ones(1, 16, dtype=torch.bool)
    # drop patch 1 and its right and lower neighbours, so patch 0 is the only
    # kept patch at distance 1
    keep[0, [1, 2, 5]] = False
    with torch.no_grad():
        out = sparse_features(vit, x, keep, fill="nearest")[-1][:, 1:]
    torch.testing.assert_close(out[0, 1], out[0, 0])


def test_blank_fill_uses_blank_tile_tokens():
    vit = tiny_vit()
    x = torch.randn(1, 6, 64, 64)
    keep = torch.ones(1, 16, dtype=torch.bool)
    keep[0, [3, 7]] = False
    with torch.no_grad():
        out = sparse_features(vit, x, keep, fill="blank")
        blank = vit.forward_features(torch.zeros_like(x))
    for o, b in zip(out, blank):
        torch.testing.assert_close(o[0, 1 + 3], b[0, 1 + 3])
        torch.testing.assert_close(o[0, 1 + 7], b[0, 1 + 7])
