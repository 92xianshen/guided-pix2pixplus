Guided-Pix2Pix+
===============

## Train
- On the image dataset:
  1. Put hazy and haze-free images with the same name in `<input>/cloud` and `<input>/label` folders
  2. Disable `fromTFRecord` in `<input>/train.xml`
  3. `python train.py <input> <output>`
- On the TFRecord dataset:
  1. Make sure `fromTFRecord` in `<input>/train.xml` is enabled
  2. `python train.py <input> <output>`

## Inference
- Please revise `inference.py` to dehaze your own test set
