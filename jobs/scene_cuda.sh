#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C 2080ti
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J geofm-scene
#SBATCH -o logs/%x-%j.out
#
# kerala flood scene (T43PFL, 25 aug 2018) on a 2080 ti: tests, scene prep,
# then terratorch vs dense vs drop. run from the repo folder on ada:
#   sbatch jobs/scene_cuda.sh
source jobs/env.sh
REPO=$PWD
cd $REPO && $PY -m pytest -q 2>&1 | tail -3

D=$SCR/scenes/T43PFL_20180825
mkdir -p $D
B="https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/43/P/FL/S2B_MSIL1C_20180825T051639_N0206_R062_T43PFL_20180825T090408.SAFE/GRANULE/L1C_T43PFL_A007667_20180825T053206/IMG_DATA/T43PFL_20180825T051639_"
for b in 01 02 03 04 05 06 07 08 8A 09 10 11 12; do
    [ -s $D/B$b.jp2 ] || curl -sf -o $D/B$b.jp2 "${B}B$b.jp2" || { echo "download B$b failed"; exit 1; }
done
[ -s $D/stack.npy ] || $PY bench/scene_prep.py $D || exit 1

mkdir -p results
$PY bench/scene_bench.py $D --device cuda --out results/scene_T43PFL_2080ti_fp32.json 2>&1 | grep -v -i warn
$PY bench/scene_bench.py $D --device cuda --fp16 --skip-terratorch --out results/scene_T43PFL_2080ti_fp16.json 2>&1 | grep -v -i warn
nvidia-smi --query-gpu=memory.used --format=csv
