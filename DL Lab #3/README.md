# ARTI 402 — Lab 3: CNN Architecture (The Four Building Blocks)

## What I did

In this lab I built the four building blocks of a convolutional neural network from scratch
using only NumPy: the filtering (convolution) layer, max pooling, flatten, and the fully
connected layer I already had from Lab 2. I then used these blocks to prove, experimentally,
that a CNN generalizes to shifted images while a plain dense network does not.

## Files in this repository

| File | Description |
|---|---|
| `arti402_Lab3_2240001433.ipynb` | My completed, fully executed notebook (all outputs included) |
| `arti402_figures.py` | Helper module that draws the diagrams used in the lab (provided, not written by me) |
| `Zebra_image.jpeg` | Sample image used in one of the architecture diagrams (provided) |

## What I implemented

- **`convolve2d`** — slides a kernel over an image with a given stride and computes the
  dot product at each position (no padding).
- **`conv_layer` / `relu_layer`** — applies a bank of filters to an image, stacks the results
  into a 3-D tensor (depth = number of filters), and applies ReLU elementwise.
- **`max_pool2d` / `maxpool_layer`** — non-overlapping max pooling on a single feature map,
  applied independently to every channel of a 3-D tensor.
- **`flatten_layer`** — unrolls a 3-D tensor into a 1-D vector.
- **`conv_params` / `dense_params`** — formulas to count the learnable parameters in a
  convolutional layer and a dense layer.
- **`extract_features`** — runs blocks 1–3 (conv → ReLU → pool → flatten) on a batch of
  images to produce a feature matrix, using a frozen bank of Sobel filters.
- **`accuracy`** — forward pass through softmax, then compares the predicted class with the
  true label.

## What I found in the assessment

I trained the same dense classification head on two different representations of the same
shifted-image dataset:

- **Raw pixels, no convolution:** 100% training accuracy, but only chance-level (33.3%)
  accuracy on the shifted test set. The model memorized *where* bright pixels were in the
  training images rather than learning what a vertical/horizontal/diagonal bar looks like.
- **CNN pipeline (frozen Sobel filters + global pooling), just 15 trainable parameters:**
  over 90% accuracy on the same shifted test set.

I also measured how the pooling tile size affects shift tolerance: larger pooling tiles
(up to fully global pooling) discard more positional detail but buy more tolerance to the
shift between train and test images, while smaller pooling tiles keep more spatial detail
but generalize worse to large shifts.

## What this taught me

- Only the filtering layer (block 1) and the fully connected layer (block 4) have learnable
  parameters — max pooling and flatten are fixed operations.
- Convolution alone does not create translation invariance; it just shifts the response
  along with the input. It is **max pooling** that actually buys shift tolerance, by
  discarding the exact position of the strongest response inside each tile.
- The right amount of pooling depends on the task: aggressive pooling helps when only
  presence/absence of a feature matters, but hurts tasks like segmentation or localization
  where the precise position of a feature is exactly what needs to be predicted.
- Freezing a filtering layer and training only a small head is the same pattern as
  transfer learning with a real pretrained CNN — it is why the assessment could show
  results with only 15 trainable parameters.

## How to run it

1. Make sure `arti402_figures.py` and `Zebra_image.jpeg` are in the same folder as the
   notebook.
2. Open `arti402_Lab3_2240001433.ipynb` and run all cells top to bottom (or
   Kernel → Restart & Run All). Every `assert` in the notebook passes.
