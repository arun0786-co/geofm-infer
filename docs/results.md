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
