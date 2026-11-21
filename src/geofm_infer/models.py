"""load the fine tuned prithvi models and run them with or without dropping.

only the sen1floods11 flood model is wired up for now. it takes 6 sentinel-2
bands (blue, green, red, nir narrow, swir 1, swir 2) as reflectance 0..1.
"""
import glob
import os

import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download

from .sparse_vit import fill_dropped, sparse_features

FLOOD_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"

# sentinel-2 L1C band order in sen1floods11 S2Hand files:
# B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12
FLOOD_S2_INDEX = [1, 2, 3, 8, 11, 12]
FLOOD_MEAN = [0.1412956, 0.13795798, 0.12353792, 0.30902815, 0.2044958, 0.11912015]
FLOOD_STD = [0.07406382, 0.07370365, 0.08692279, 0.11798815, 0.09772074, 0.07659938]


def load_flood_model(device="cpu"):
    from terratorch.models import EncoderDecoderFactory

    root = snapshot_download(FLOOD_REPO)
    cfg = yaml.safe_load(open(os.path.join(root, "config.yaml")))
    args = dict(cfg["model"]["init_args"]["model_args"])
    args["backbone_pretrained"] = False  # weights come from the checkpoint
    model = EncoderDecoderFactory().build_model(task="segmentation", **args)

    ckpt = torch.load(glob.glob(os.path.join(root, "*.pt"))[0], map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    if str(device).startswith("mps"):
        _fix_pool_on_mps(model)
    return model.eval().to(device)


class _CpuAdaptivePool(torch.nn.Module):
    """mps can't do adaptive pooling when sizes don't divide. the upernet
    pyramid pools to 1, 2, 3 and 6, so do that one small op on the cpu."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.cpu(), self.size).to(x.device)


def _fix_pool_on_mps(model):
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.AdaptiveAvgPool2d):
                setattr(mod, child_name, _CpuAdaptivePool(child.output_size))


def normalize(x, mean=FLOOD_MEAN, std=FLOOD_STD):
    """x: (b, c, h, w) reflectance."""
    m = torch.tensor(mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    s = torch.tensor(std, device=x.device, dtype=x.dtype)[None, :, None, None]
    return (x - m) / s


@torch.no_grad()
def segment(model, x, keep=None, fill="mean", dense_encoder=False):
    """logits (b, classes, h, w). keep=None runs the normal dense model.

    dense_encoder=True runs the full encoder and only fills the dropped places
    before the decoder (a diagnostic, see fill_dropped)."""
    size = x.shape[-2:]
    if keep is None:
        feats = model.encoder(x)
    elif dense_encoder:
        feats = fill_dropped(model.encoder(x), keep, fill=fill)
    else:
        feats = sparse_features(model.encoder, x, keep, fill=fill)
    feats = model.neck(feats, image_size=size)
    out = model.head(model.decoder([f for f in feats]))
    if out.shape[-2:] != size:
        out = F.interpolate(out, size=size, mode="bilinear")
    return out
