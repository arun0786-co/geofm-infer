# data

nothing here is in git. everything is public and free to download.

## sen1floods11 (hand labelled chips)

bonafilia et al. 2020, cvpr workshops. google cloud bucket `sen1floods11`, no login.

```
D=~/geofm-infer-data/sen1floods11; mkdir -p $D/S2Hand $D/LabelHand
G=https://storage.googleapis.com/sen1floods11/v1.1
curl -o $D/flood_test_data.csv $G/splits/flood_handlabeled/flood_test_data.csv
for id in $(cut -d, -f1 $D/flood_test_data.csv | sed 's/_S1Hand.tif//'); do
  curl -so $D/S2Hand/${id}_S2Hand.tif $G/data/flood_events/HandLabeled/S2Hand/${id}_S2Hand.tif
  curl -so $D/LabelHand/${id}_LabelHand.tif $G/data/flood_events/HandLabeled/LabelHand/${id}_LabelHand.tif
done
```

S2Hand files are sentinel-2 L1C, 13 bands, 512 x 512. labels: 1 water, 0 not water, -1 no label.

## sentinel-2 scene

from google's public copy of sentinel-2 L1C (`gcp-public-data-sentinel-2`), because the aws
copy is requester pays. tile 43PFL, 2018-08-25, kerala floods. the 13 band urls are in
`jobs/scene_cuda.sh`. about 240 MB.

## models

`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` on hugging face (1.3 GB),
downloaded on first use.
