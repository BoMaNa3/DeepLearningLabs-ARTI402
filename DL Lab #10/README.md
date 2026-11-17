# Lab 10 — Attention

Student ID: 2240001433

I implemented scaled dot-product attention using a stable softmax, tested exact and blended lookups, measured the effect of scaling, compared retrieval with a summary-vector bottleneck, and implemented self-attention and a causal mask.

I completed both assessment questions and recorded the actual attention weights in the notebook. At sequence length 64, I measured attention accuracy of 0.9575, compared with 0.0150 for the bottleneck. I verified that causal attention assigns zero weight to future positions and that the first position attends only to itself. I used a sorted vocabulary so the random word embeddings are reproducible across fresh Python processes.

## How I run my notebook

I keep `arti402_Lab10_2240001433.ipynb`, `arti402_figures.py`, and `requirements.txt` in the same folder. I open a terminal in that folder and prepare my Python environment with:

```powershell
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyterlab
```

I open the completed notebook, select the Python kernel for that environment, and choose **Restart Kernel and Run All Cells**. The notebook also installs missing core dependencies into its current kernel if needed; I need internet for package installation. I run cells in order and keep all existing outputs visible in the submitted file.

## My verification

I ran the entire notebook in a fresh Python 3.12.10 Jupyter kernel and passed every assertion, with all figures and results retained. My tested packages are pinned in `requirements.txt`. Results can differ slightly across numerical-library versions or hardware. I have included the original course figure helper because the notebook imports it.
