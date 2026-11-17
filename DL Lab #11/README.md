# Lab 11 — Generative Models: VAEs and GANs

Student ID: 2240001433

I implemented reparameterization, Gaussian KL divergence, the full VAE loss, numerical gradient checking, random digit generation, a classifier-based generation score, latent interpolation, and the discriminator and generator gradients for a one-dimensional GAN. I answered the checkpoint and completed both assessment tables using actual training results.

I kept the specified 400 training epochs, random seeds, data split, and GAN training settings. I reused models already trained at the same beta and seed to avoid repeating identical training. The digits dataset is bundled with scikit-learn, so I do not download a dataset.

I measured a worst numerical-gradient relative error of 1.98e-06. My digit judge achieved 96.889% accuracy on held-out real digits. I obtained generation confidence 0.4990 for beta=0 and 0.6759 for beta=1. I selected beta=1 because it gave the best reconstruction loss and generation confidence among the positive-beta settings I tested, and I bracketed the practical collapse between beta=2 and beta=3.

My GAN ended with generated mean 2.9804 and standard deviation 0.5215, close to the target mean 3 and standard deviation 0.5. The final discriminator averages were 0.5143 on real samples and 0.4957 on generated samples. I retained the imperfect interpolation output (0, 0, 0, 8, 8, 8, 8, 1) and treated judge confidence as a proxy rather than a guarantee of image quality or diversity.

## How I run my notebook

I keep `arti402_Lab11_2240001433.ipynb`, `arti402_figures.py`, and `requirements.txt` in the same folder. I open a terminal in that folder and prepare my Python environment with:

```powershell
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyterlab
```

I open the completed notebook, select the Python kernel for that environment, and choose **Restart Kernel and Run All Cells**. The notebook also installs missing core dependencies into its current kernel if needed; I need internet for package installation. I run cells in order and keep all existing outputs visible in the submitted file.

## My verification

I ran the entire notebook in a fresh Python 3.12.10 Jupyter kernel and passed every assertion, with all figures and results retained. My tested packages are pinned in `requirements.txt`. Lab 11 also uses scikit-learn 1.5.2 and one BLAS thread for its small matrix calculations. Results can differ slightly across numerical-library versions or hardware. I have included the original course figure helper because the notebook imports it.
