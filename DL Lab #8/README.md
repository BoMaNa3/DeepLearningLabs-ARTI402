# My ARTI402 Lab 8 Solution

ARTI402 Deep Learning  
Maan Abdullah Alghamdi  
2240001433

I completed the six exercises and both checkpoints in the supplied PyTorch notebook. I added my course, name, and student ID immediately after the lab title and saved the executed notebook as `arti402_Lab8_2240001433.ipynb`. I omitted the angle brackets because Windows does not allow them in filenames.

## What I implemented

- I checked autograd against a central finite-difference gradient in float64.
- I built the 2 → 128 → 3 network with ReLU and optional dropout, returning logits for cross-entropy.
- I wrote the training loop with gradient clearing, a forward pass, backpropagation, and an Adam update.
- I evaluated loss and accuracy with evaluation mode and gradient recording disabled.
- I implemented batching with a seeded DataLoader and a sample-weighted mean loss.
- I completed the reusable training function with dropout, weight decay, and training/validation history.
- I answered both checkpoints, including why clearing gradients after each update can also be valid when the gradients start clear.

The supplied notebook refers to an assessment, but it ends after Exercise 6 and the recap. I completed every exercise actually included; I did not invent a missing assessment.

## My files

- `arti402_Lab8_2240001433.ipynb`: my completed notebook with execution outputs saved.
- `arti402_figures.py`: the original figure helper, which I keep beside the notebook.
- `requirements.txt`: the package versions used for verification.
- `README.md`: my explanation and rerun instructions.

## How I run my notebook

I use Python 3.12 on Windows. From PowerShell in this lab folder, I create a dedicated environment and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook arti402_Lab8_2240001433.ipynb
```

I select the Python kernel from this environment and choose **Restart Kernel and Run All Cells**. I run the cells from top to bottom because later exercises use functions defined earlier. I keep the helper file in this folder. The first dependency installation requires internet access; the lab itself generates its spiral data locally and needs no GPU or external dataset.

If I use VS Code instead, I select `.venv\Scripts\python.exe` as my notebook kernel before running all cells. I do not need to activate the environment or change PowerShell execution policy to use the commands above.

I use fixed seeds and one CPU thread for this small network. Timing may differ between computers, and numerical results can vary slightly across different package versions or hardware.

## My verification results

I ran all 11 code cells in order from a fresh Python 3.12.10 kernel with PyTorch 2.14.1+cpu. Every supplied assertion passed, and the saved notebook contains no error outputs.

- My maximum gradient-check difference was `3.98e-11`, below the required `1e-8`.
- My network contained exactly 771 parameters.
- My manual training loop reduced the loss from `1.201` at epoch 1 to `1.003` at epoch 10.
- My DataLoader produced four batches per epoch and a loss of `0.996` after ten epochs.
- My 100-epoch run recorded 21 history rows and passed the training-accuracy improvement check.

These are the supplied short training checks, not a claim of high final classification accuracy. I saved all outputs produced by the provided exercises; the source notebook does not call any plotting functions.
