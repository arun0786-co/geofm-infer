# results

all numbers come from json files in `results/`, made by the commands shown. nothing here
is estimated. failed ideas stay in, with their numbers.

## 1. sen1floods11 test chips, m4 pro (mps), 2026-11-21

```
python bench/cloud_masks.py test
python bench/eval_sen1floods11.py --device mps --modes clean,zero_fill,drop_mean,drop_near,dec_mean \
    --out results/eval_test_mps_v2.json
python bench/eval_sen1floods11.py --device mps --modes clean,zero_fill,drop_mean,drop_blank \
    --out results/eval_test_mps_blank.json
```

model: Prithvi-EO-2.0-300M-TL-Sen1Floods11 (upernet decoder, 319 M parameters), fp32.
90 hand labelled test chips of 512 x 512, one chip per forward pass (batch 1).
scores are water iou and accuracy on labelled pixels that are **not** hidden.
"agree" is the share of those pixels where the answer equals the clean run.
in every mode except clean, hidden pixels are blanked (set to the band mean).

| hidden | mode | water iou | acc | agree w/ clean | tokens run | ms / chip |
|---|---|---|---|---|---|---|
| 0% | clean | 0.8168 | 0.9754 | 1.0000 | 1.00 | 211 |
| | drop_mean (nothing dropped) | 0.8168 | 0.9754 | 1.0000 | 1.00 | 235 |
| 25% blobs | zero_fill | 0.8233 | 0.9763 | 0.9930 | 1.00 | 212 |
| | drop_mean | 0.8074 | 0.9744 | 0.9936 | 0.75 | 185 |
| | drop_near | 0.8064 | 0.9741 | 0.9935 | 0.75 | 196 |
| | drop_blank | 0.7846 | 0.9714 | 0.9906 | 0.75 | 192 |
| | dec_mean (check) | 0.8170 | 0.9755 | 0.9913 | 0.75 | 248 |
| 50% blobs | zero_fill | 0.8232 | 0.9761 | 0.9879 | 1.00 | 213 |
| | drop_mean | 0.7949 | 0.9727 | 0.9891 | 0.50 | 138 |
| | drop_near | 0.7939 | 0.9724 | 0.9896 | 0.50 | 149 |
| | drop_blank | 0.7528 | 0.9675 | 0.9842 | 0.50 | 142 |
| | dec_mean (check) | 0.8065 | 0.9738 | 0.9843 | 0.50 | 247 |
| 75% blobs | zero_fill | 0.8093 | 0.9734 | 0.9823 | 1.00 | 212 |
| | drop_mean | 0.7755 | 0.9696 | 0.9816 | 0.25 | 95 |
| | drop_near | 0.7778 | 0.9695 | 0.9828 | 0.25 | 106 |
| | drop_blank | 0.6972 | 0.9601 | 0.9733 | 0.25 | 96 |
| | dec_mean (check) | 0.7715 | 0.9680 | 0.9753 | 0.25 | 246 |
| real clouds (s2cloudless) | zero_fill | 0.8503 | 0.9783 | 0.9954 | 1.00 | 212 |
| | drop_mean | 0.8485 | 0.9782 | 0.9970 | 0.80 | 196 |
| | drop_near | 0.8456 | 0.9778 | 0.9972 | 0.80 | 206 |
| | drop_blank | 0.8408 | 0.9771 | 0.9970 | 0.80 | 202 |
| | dec_mean (check) | 0.8512 | 0.9784 | 0.9950 | 0.80 | 242 |

the modes:

- zero_fill: blank the hidden pixels, run the normal model. the baseline to beat.
- drop_*: also blank them, and patches that are at least half hidden never enter the
  encoder. after the encoder the dropped places get a fill: the mean of the kept tokens
  (mean), a copy of the closest kept patch (near), or the encoder's output for an all
  zero tile (blank).
- dec_mean: not a speed up, a check. the full encoder runs on the zero_fill image and only
  then the hidden places are overwritten with the mean fill. if this loses as much as
  drop_mean, the loss comes from the decoder seeing fill tokens, not from the encoder
  missing context.

what this says:

- **real clouds: dropping is close to free in accuracy** (0.8485 vs 0.8503 iou, 0.9970
  vs 0.9954 agreement) **but saves only 8% time**, because 37 of 90 chips have more than
  5% cloud and few whole patches are covered (20% cloud on average, 20% of tokens dropped).
- big hidden areas: time falls with the tokens dropped (212 to 95 ms at 75%, 2.2x), and
  water iou falls by 1.6, 2.8 and 3.4 points at 25, 50 and 75%. a trade, not free.
- most of that loss is in the decoder. dec_mean, which gives the encoder full context,
  still loses 0.6, 1.7 and 3.8 points. so the upernet decoder doesn't like fill tokens
  it never saw in training. the encoder missing context explains less than half the loss.
- the mean fill is the best of the three fills. **the blank fill was my idea for fixing
  the decoder side and it is clearly worse** (0.6972 at 75%). the encoder's tokens for a
  blank tile don't look like the tokens for blanked pixels inside a real scene.
- the sparse path costs 10 to 15% when nothing is dropped (235 vs 211 ms), from gathering
  and scattering tokens. so it should only be used when enough is dropped.
- zero_fill beats clean at 25 and 50%. not explained yet.

the first run (`results/eval_test_mps.json`) had a bug in the comparison: drop modes did
not blank the hidden pixels inside kept patches, so they saw pixels zero_fill did not.
its numbers are not used here.

next: a short fine tune of the decoder with random drops, so it learns the mean fill.

## 2. a whole sentinel-2 scene, m4 pro (mps), 2026-11-21

```
python bench/scene_prep.py ~/geofm-infer-data/scenes/T43PFL_20180825
python bench/scene_bench.py ~/geofm-infer-data/scenes/T43PFL_20180825 --device mps --batch 1 \
    --out results/scene_T43PFL_m4pro_fp32_b1.json
```

tile 43PFL, 25 aug 2018 (kerala floods), 10980 x 10980 px at 10 m. 66.5% of the tile is
outside the satellite's swath (nodata), and s2cloudless marks 24% of the rest as cloud, so
74% is not clear. 512 px tiles with 64 px overlap: 625 tiles. same model as section 1, fp32.

| run | seconds | tiles run | tokens run | agree with dense (clear px) | water share (clear px) |
|---|---|---|---|---|---|
| terratorch tiled_inference | 140.2 | 625 | all | 0.9269 | 0.2615 |
| dense (our tiler, everything) | 142.7 | 625 | 640,000 | 1 | 0.2622 |
| skip (empty tiles left out) | **50.9** | 194 | 198,656 | **1.0000** | 0.2622 |
| drop (skip + drop cloudy patches) | **42.3** | 194 | 159,623 | 0.9439 | 0.2357 |

what this says:

- **skipping tiles with no clear patch is the big, free win on this scene: 2.75x faster than
  terratorch with exactly the same answer on every clear pixel.** most of it is the nodata
  part of the tile. many sentinel-2 tiles at a swath edge look like this, but a tile in the
  middle of a swath would gain only from clouds.
- dropping cloudy patches inside the tiles that do run saves another 17% (50.9 to 42.3 s),
  but changes 5.6% of the clear pixels and finds less water (23.6% against 26.2%). on the
  labelled chips (section 1) the same kind of change cost little iou, but here there is no
  label to say which answer is right.
- terratorch and our dense tiler disagree on 7.3% of clear pixels although both run every
  tile with the same model. they differ only in how tiles are cut, padded and blended. so
  tiling choices alone move this many pixels on this model, which is worth its own look.
- batch 1 on purpose: on mps, batches of tiles that keep different numbers of patches need
  padding and an attention mask, and that was slower than one tile at a time (crop test:
  drop 2.15 s at batch 1, 2.68 s at batch 8). batch size barely changed dense times.

## 3. the same scene on an rtx 2080 ti (cuda), 2026-11-21

`sbatch jobs/scene_cuda.sh` (ada cluster, iiit hyderabad). batch 8, everything else as in
section 2. results: `results/scene_T43PFL_2080ti_fp32.json`, `..._fp16.json`.

| run | fp32 s | fp16 s | agree with dense (clear px, fp32) |
|---|---|---|---|
| terratorch tiled_inference | 64.1 | not run | 0.9269 |
| dense (our tiler) | 66.5 | 20.8 | 1 |
| skip | **22.7** | **7.8** | **1.0000** |
| drop | 20.3 | 7.6 | 0.9439 |

- same picture as on the mac: skipping empty tiles is 2.8x faster than terratorch (fp32 vs
  fp32) with the same answer.
- fp16 makes everything about 3x faster again on this card (tensor cores). i did not run
  terratorch in fp16, so "7.8 s vs 64.1 s" would not be a fair comparison and i don't claim it.
- dropping patches on top of skipping saves only 10% (fp32) and 2% (fp16) here, less than on
  the mac. tiles in a batch of 8 are padded to the longest one and need an attention mask,
  which eats most of the saving. better packing (no padding) is the obvious next thing.
- the water share and agreement numbers are the same as on the mac to 3 or 4 digits, so the
  mps and cuda runs compute the same thing.

## 4. drop aware decoder fine tune, rtx 2080 ti, 2026-11-21/22

`sbatch --export=ALL,SEED=<0|1|2> jobs/train_drop.sh`. encoder frozen; neck, decoder and
head (15.1 M parameters) trained 10 epochs on the 252 sen1floods11 training chips, lr 1e-5,
batch 4, batch norm statistics frozen. each chip gets a random hidden area of 0 to 80% (0%
for one chip in five), hidden pixels blanked, patches more than half hidden dropped with
the mean fill. 34 s per epoch. then the section 1 eval on the 90 test chips
(`results/eval_test_2080ti_orig.json`, `..._tuned_s0/1/2.json`).

the original model gives the same numbers on cuda as on mps (section 1) to 4 digits.

water iou; tuned = mean of 3 seeds, with the lowest and highest seed:

| hidden | mode | original | tuned (3 seeds) |
|---|---|---|---|
| 0% | clean | 0.8168 | 0.8181 (0.8159 to 0.8194) |
| 25% | zero_fill | 0.8233 | 0.8153 (0.8126 to 0.8180) |
| | drop_mean | 0.8074 | 0.8009 (0.7975 to 0.8026) |
| 50% | zero_fill | 0.8232 | 0.8179 (0.8156 to 0.8191) |
| | drop_mean | 0.7949 | 0.7941 (0.7920 to 0.7953) |
| 75% | zero_fill | 0.8093 | 0.7989 (0.7907 to 0.8079) |
| | drop_mean | 0.7755 | **0.7858 (0.7844 to 0.7868)** |
| real clouds | zero_fill | 0.8503 | 0.8524 (0.8512 to 0.8539) |
| | drop_mean | 0.8485 | **0.8522 (0.8517 to 0.8526)** |

(ms per chip on the 2080 ti, batch 1: 92 dense, 78 / 59 / 37 for drop at 25 / 50 / 75%.)

what this says:

- **with real clouds, the tuned model's drop run (0.8522) is as good as the tuned model's
  full run (0.8524) and better than the original full run (0.8503)**, in every seed, while
  leaving the cloudy patches out of the encoder. this is the useful result.
- at 75% hidden, every seed improves drop_mean by about 1 point (0.7755 to 0.7858). the
  original model's zero_fill (0.8093) is still better there.
- at 25% hidden the tuned model is slightly worse in every seed (0.8074 to 0.8009). at 50%
  nothing changes.
- the normal dense use is not hurt (0.8168 to 0.8181).
- the tuned model got worse at zero_fill with synthetic holes. it was never trained that
  way, so it is a model for the drop path, not a replacement.
