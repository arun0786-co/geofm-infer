# related work

what already exists, and where this repo is different. checked 2026-11-21.

- **terratorch tiled inference** ([docs](https://terrastackai.github.io/terratorch/1.2.1/guide/inference)).
  cuts a big image into crops with a stride and a `delta` border that is thrown away, then
  averages the overlaps. every crop runs through the full model, clouds and nodata included.
  this is the baseline in `bench/scene_bench.py`.
- **geoserve** ([ibm research, euromlsys 2026](https://research.ibm.com/publications/geoserve-leveraging-disaggregated-data-processing-for-scalable-geospatial-model-serving)).
  a serving system: splits raster reading and preprocessing from the gpu work, and batches
  requests. it reports that reading and preprocessing take 31 to 43% of request time. it is
  about many requests on a server, not about skipping work inside one image.
- **switi** ([arxiv 2607.18990](https://arxiv.org/pdf/2607.18990)). measures and reduces tiling
  artifacts (seams) with sliding window inner tiling. useful for measuring seams here.
- **masked autoencoders** (he et al. 2022, [arxiv 2111.06377](https://arxiv.org/abs/2111.06377)).
  the encoder is trained on only the visible 25% of patches. prithvi (eo 1.0 and 2.0) is
  pretrained this way, which is why dropping patches at inference is not a crazy idea.
- **token pruning / merging for vits** (e.g. tome, [arxiv 2210.09461](https://arxiv.org/abs/2210.09461)).
  these learn or compute which tokens to drop from the image content. here the tokens to drop
  are known before the model runs (cloud and nodata masks), so no extra model is needed.

what i did not find: anyone dropping cloudy or nodata patches before the encoder for
geospatial foundation models, and measuring what it costs in accuracy on the clear part.
my search was not exhaustive.
