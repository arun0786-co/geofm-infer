#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J geofm-drop
#SBATCH -o logs/%x-%j.out
#
# drop aware decoder fine tune on sen1floods11, then the same eval as on the
# mac for the original and the tuned decoder.   sbatch jobs/train_drop.sh
source jobs/env.sh
export GEOFM_DATA=$SCR/sen1floods11
D=$GEOFM_DATA
mkdir -p $D/S2Hand $D/LabelHand
G=https://storage.googleapis.com/sen1floods11/v1.1
for s in train valid test; do
    [ -s $D/flood_${s}_data.csv ] || curl -sf -o $D/flood_${s}_data.csv $G/splits/flood_handlabeled/flood_${s}_data.csv
    for id in $(cut -d, -f1 $D/flood_${s}_data.csv | sed 's/_S1Hand.tif//'); do
        [ -s $D/S2Hand/${id}_S2Hand.tif ] || curl -sf -o $D/S2Hand/${id}_S2Hand.tif $G/data/flood_events/HandLabeled/S2Hand/${id}_S2Hand.tif
        [ -s $D/LabelHand/${id}_LabelHand.tif ] || curl -sf -o $D/LabelHand/${id}_LabelHand.tif $G/data/flood_events/HandLabeled/LabelHand/${id}_LabelHand.tif
    done
done
echo "chips: $(ls $D/S2Hand | wc -l)"
[ -d $D/clouds ] || $PY bench/cloud_masks.py test > /dev/null

mkdir -p runs results
SEED=${SEED:-0}
$PY bench/train_drop.py --device cuda --epochs ${EPOCHS:-10} --seed $SEED --out runs/drop_decoder_s$SEED.pt 2>&1 | grep -v -i warn
M=clean,zero_fill,drop_mean
$PY bench/eval_sen1floods11.py --device cuda --modes $M --out results/eval_test_2080ti_orig.json 2>&1 | grep -v -i warn | tail -26
$PY bench/eval_sen1floods11.py --device cuda --modes $M --decoder runs/drop_decoder_s$SEED.pt \
    --out results/eval_test_2080ti_tuned_s$SEED.json 2>&1 | grep -v -i warn | tail -26
