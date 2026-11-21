# results

all numbers come from files in `results/`, made by the commands shown. nothing here is
estimated.

## 1. sen1floods11 test chips, m4 pro (mps), 2026-11-21

`python bench/cloud_masks.py test && python bench/eval_sen1floods11.py --device mps --out results/eval_test_mps.json`

model: Prithvi-EO-2.0-300M-TL-Sen1Floods11 (upernet decoder, 319 M parameters), fp32.
90 hand labelled test chips of 512 x 512. one chip per forward pass (batch 1).
scores are water iou and accuracy on labelled pixels that are **not** hidden.
"agree" is the share of those pixels where the answer equals the clean run.

| hidden | mode | water iou | acc | agree w/ clean | tokens run | ms / chip |
|---|---|---|---|---|---|---|
| 0% | clean | 0.8168 | 0.9754 | 1.0000 | 1.00 | 210 |
| 25% blobs | zero_fill | 0.8233 | 0.9763 | 0.9930 | 1.00 | 210 |
| | drop_mean | 0.8059 | 0.9742 | 0.9956 | 0.75 | 184 |
| | drop_near | 0.8059 | 0.9741 | 0.9958 | 0.75 | 194 |
| 50% blobs | zero_fill | 0.8232 | 0.9761 | 0.9879 | 1.00 | 210 |
| | drop_mean | 0.7959 | 0.9728 | 0.9915 | 0.50 | 137 |
| | drop_near | 0.7977 | 0.9730 | 0.9928 | 0.50 | 148 |
| 75% blobs | zero_fill | 0.8093 | 0.9734 | 0.9823 | 1.00 | 210 |
| | drop_mean | 0.7807 | 0.9701 | 0.9855 | 0.25 | 94 |
| | drop_near | 0.7854 | 0.9706 | 0.9881 | 0.25 | 105 |
| real clouds (s2cloudless) | zero_fill | 0.8503 | 0.9783 | 0.9954 | 1.00 | 209 |
| | drop_mean | 0.8414 | 0.9772 | 0.9975 | 0.80 | 194 |
| | drop_near | 0.8387 | 0.9768 | 0.9976 | 0.80 | 204 |

drop_zero was worst of the drop modes everywhere (0.7266 iou at 75%), see the json.
the "clean" row changes a little between settings because the scored pixels change.

what this says:

- time goes down with the share of patches dropped: 210 ms to 94 ms at 75% (2.2x).
  at 0% dropped the sparse path costs 10 to 15% extra (233 vs 210 ms), from gathering
  and scattering tokens.
- dropping costs accuracy. against zero_fill it loses 1.7 iou points at 25%, 2.6 at 50%
  and 2.4 at 75%. so on this model dropping is a speed for accuracy trade, not free.
- dropped runs agree with the clean run more often than zero_fill does (0.9956 vs 0.9930
  at 25%), but their water iou is lower. they change fewer pixels, but the pixels they
  change are more often water.
- zero_fill beats clean at 25 and 50%. i don't have an explanation yet; it may be
  that hiding pixels removes some confusing context near water. not checked.
- real clouds: 37 of 90 chips have more than 5% cloud, 20% on average. here the drop
  modes lose about 1 iou point and save only 7% time, because few whole patches are cloudy.

open question: does the loss come from the encoder seeing less context, or from the
decoder getting fill tokens it never saw in training? next experiment separates them.
