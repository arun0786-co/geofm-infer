# geofm-infer

run geospatial foundation models (prithvi) over whole satellite scenes, and skip the work
that can't give an answer: tiles that are all cloud or outside the image, and inside a
tile, the patches that are cloudy.

prithvi was pretrained as a masked autoencoder, so its encoder already works with part of
the image missing. here that is used at inference: cloudy and nodata patches never go into
the transformer. the outputs are put back on the full grid with a fill value, so the normal
decoder runs unchanged.

this is a measurement project as much as a tool. every number below comes from a json file
in `results/` and a command in `docs/results.md`, and the results include what didn't work.

## what i found so far

on the 90 hand labelled sen1floods11 test chips with the official
Prithvi-EO-2.0-300M flood model (m4 pro):

- **with real clouds, dropping cloudy patches costs almost nothing**: water iou 0.8485
  against 0.8503 for the usual way (blank the cloudy pixels and run everything). but it only
  saves 8% time there, because most clouds don't cover whole patches.
- **with big hidden areas it is a trade**: 2.2x faster at 75% hidden, but 3.4 points less
  water iou.
- **the loss is mostly in the decoder**, not in the encoder missing context. a check that
  gives the encoder full context and only fills the decoder input loses about as much.
- two ideas that did not help: copying the nearest kept patch (same as the mean), and using
  the encoder's tokens for a blank tile as the fill (clearly worse).

on a whole sentinel-2 scene (kerala floods, 25 aug 2018, two thirds outside the swath):
see `docs/results.md`, section 2. (being measured)

## how it works

1. `tokens.py`: a pixel cloud / nodata mask becomes a per patch keep mask (a patch is kept
   when at least half of it is clear).
2. `sparse_vit.py`: kept patches are moved to the front, tiles in a batch are padded to the
   same length with an attention mask, the prithvi blocks run on that, and the results are
   scattered back to the full grid with a fill (mean of the kept tokens by default).
3. `scene.py`: the scene is cut into overlapping 512 px tiles. tiles with no clear patch are
   skipped, the rest are sorted by how many patches they keep (less padding), and the
   overlaps are blended with a weight that is low at tile edges.

## run it

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e . pytest s2cloudless pystac-client
.venv/bin/python -m pytest -q

# sen1floods11 test chips (downloads in docs/data.md)
.venv/bin/python bench/cloud_masks.py test
.venv/bin/python bench/eval_sen1floods11.py --device mps

# a whole scene
.venv/bin/python bench/scene_prep.py ~/geofm-infer-data/scenes/T43PFL_20180825
.venv/bin/python bench/scene_bench.py ~/geofm-infer-data/scenes/T43PFL_20180825 --device mps --batch 1
```

`jobs/` has the slurm scripts used for the cuda numbers (rtx 2080 ti).

## limits

- only the sen1floods11 flood model is wired up so far.
- the cloud mask is s2cloudless with default settings. a bad mask means bad drops.
- the sparse path is 10 to 15% slower than the normal one when nothing is dropped.
- there is no labelled whole scene, so on scenes i can only compare against the dense run,
  not against the truth.

## related work

see `docs/related_work.md`. the closest are terratorch's tiled inference (the baseline here)
and ibm's geoserve (serving, not skipping work inside an image).
