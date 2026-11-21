import numpy as np
import torch

from geofm_infer.scene import Plan, blend_weight, run_scene, tile_starts


def test_tile_starts_cover():
    s = tile_starts(1000, 512, 64)
    assert s[0] == 0 and s[-1] == 1000 - 512
    assert all(b - a <= 512 - 64 for a, b in zip(s, s[1:]))
    assert tile_starts(300, 512, 64) == [0]


def test_blend_weight_shape():
    w = blend_weight(64, 16)
    assert w.shape == (64, 64) and w.max() == 1.0 and w.min() > 0


def fake_model(x, keep):
    # class 1 where the first band is positive
    p = (x[:, :1] > 0).float()
    return torch.cat([1 - p, p], 1) * 10


def test_run_scene_stitches_and_skips():
    img = np.random.randn(2, 200, 300).astype(np.float32)
    clear = np.ones((200, 300), bool)
    clear[:, :112] = False            # left part is all cloud
    pred, st = run_scene(fake_model, img, clear, Plan(tile=64, overlap=16, patch=16))
    want = (img[0] > 0).astype(np.int16)
    # inside the clear part the stitched answer equals the per pixel answer
    assert (pred[:, 128:] == want[:, 128:]).all()
    assert (pred[:, :96] == -1).all()  # tiles that were all cloud never ran
    assert st["tiles_run"] < st["tiles"]
