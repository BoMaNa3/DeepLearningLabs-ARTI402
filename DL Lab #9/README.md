# My ARTI 402 Lab 9 — Autoencoders

**Student ID:** 2240001433  
**Notebook:** `arti402_Lab9_2240001433.ipynb`

## What I completed

In this lab, I explored how autoencoders learn representations without digit labels. I completed all six coding exercises and answered both checkpoints in the notebook. I built the encoder and decoder, implemented the reconstruction training loop, compared bottleneck sizes, plotted a 2-D latent space, measured code quality with logistic regression, and trained a denoising autoencoder.

## My method

I loaded scikit-learn's 8 × 8 handwritten digits and divided the pixel values by 16 to scale them to [0, 1]. I used a stratified 75%/25% split with `random_state=0`, giving me 1,347 training images and 450 test images.

My main network followed `64 → 32 → 8 → 32 → 64`. I used ReLU in the hidden layers, a linear latent code, and Sigmoid at the output. I trained with mean squared reconstruction error, Adam, a learning rate of 0.001, batch size 64, and 60 epochs. I set the model and loader seeds to 0 for reproducibility and used one CPU thread for these small networks.

## My measured results

My 8-number model's training MSE fell from **0.178206** to **0.021560**, and its test MSE was **0.021908**.

| Code dimensions | Dimension ratio | Train MSE | Test MSE |
|---:|---:|---:|---:|
| 2 | 32× | 0.046671 | 0.047906 |
| 4 | 16× | 0.034592 | 0.036816 |
| 8 | 8× | 0.021278 | 0.021908 |
| 16 | 4× | 0.019306 | 0.020321 |
| 32 | 2× | 0.017920 | 0.018955 |

I used the training-set average image as a simple baseline; its test MSE was **0.074027**. I interpret the compression ratios as reductions in representation dimensions, rather than as actual file compression ratios.

| Logistic regression input | Test accuracy |
|---|---:|
| My 2-number codes | 62.22% |
| My 8-number codes | 92.44% |
| All 64 pixels | 96.89% |

I found that eight latent coordinates retained more class information than two. The encoder learned these features from pixel reconstruction; I introduced digit labels only for the scatter plot and supervised logistic regression comparisons.

For denoising, I corrupted each training batch with Gaussian noise of standard deviation 0.5 and clamped it to [0, 1]. I kept the clean images as targets. I generated one noisy test tensor using seed 7 and used it for both models.

| Model | Clean-input MSE | Noisy-input MSE against clean images |
|---|---:|---:|
| My plain autoencoder | 0.021908 | 0.053449 |
| My denoising autoencoder | 0.031546 | 0.043580 |
| My unrepaired noisy input | — | 0.115681 |

## What I learned

I learned that reconstruction pressure can produce useful features without human labels. A smaller bottleneck forces stronger compression but loses more detail. I also learned why a denoiser must compare its output against the clean target: using the noisy target would reward reproducing the noise.

In Checkpoint 1, I clarified that changing only the latent layer to 64 units does not guarantee zero loss in this architecture: the hidden layers still have 32 units, and a finite sigmoid cannot exactly reproduce 0 or 1. I also distinguished a valid supervised classifier from directly passing integer labels into an incompatible reconstruction loss.

## How I run my notebook

1. I keep `arti402_figures.py` in the same folder as my notebook.
2. In a Python environment with Jupyter support, I install any missing dependencies with `pip install torch scikit-learn matplotlib ipykernel`.
3. I open the notebook, choose that Python kernel, and use **Restart Kernel and Run All**.

The digits dataset is bundled with scikit-learn, so I do not need to download a separate dataset. My saved notebook already contains the executed outputs and plots. I verified it with a full top-to-bottom run: all original assertions passed, and 6 plots were embedded. Exact values may vary slightly with package versions or hardware. The original lab's PCA comparison is marked as illustrative; I did not claim to have run that extra experiment.

## My files

- `arti402_Lab9_2240001433.ipynb`: my completed and executed lab, including checkpoint answers and measured results.
- `arti402_figures.py`: the supplied plotting helper, copied unchanged.
- `README.md`: this first-person explanation of my method, results, and execution steps.
