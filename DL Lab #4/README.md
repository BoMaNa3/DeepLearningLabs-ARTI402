# ARTI 402 — Lab 4: Recurrent Networks

This repository contains my completed **ARTI 402 Lab 4**, where I implemented and compared recurrent neural-network architectures using NumPy.

## What I implemented

In this lab, I:

- implemented one forward step of a vanilla RNN;
- unrolled the RNN over sequences of different lengths;
- calculated the parameter counts for RNN, GRU, and LSTM cells;
- implemented the forget, input, candidate, and output operations of an LSTM cell;
- extracted final hidden states from RNN and LSTM models;
- trained dense classification heads on the frozen recurrent features;
- tested how well each model remembered information from the first time step; and
- investigated how the LSTM forget-gate bias affects long-term memory.

## My results

The experiment showed that the plain RNN quickly lost the information provided at the beginning of a sequence, while the LSTM retained it for much longer.

| Sequence length | RNN accuracy | LSTM accuracy |
|---:|---:|---:|
| 2 | 1.000 | 1.000 |
| 5 | 0.608 | 0.992 |
| 10 | 0.292 | 1.000 |
| 20 | 0.317 | 0.983 |
| 40 | 0.275 | 0.942 |
| 80 | 0.333 | 0.792 |

These results helped me see the vanishing-memory problem directly. Repeated recurrent transformations caused the RNN's early signal to disappear, whereas the LSTM's cell state and forget gate provided a more stable path for carrying information forward.

## Files

- `arti402_Lab4_2240001433.ipynb` — my completed and executed Lab 4 notebook.
- `arti402_figures.py` — the supplied helper module used to generate the diagrams. This file should be placed in the same folder as the notebook when rerunning it.

## How I run the notebook

1. I place `arti402_Lab4_2240001433.ipynb` and `arti402_figures.py` in the same directory.
2. I install the required packages:

   ```bash
   pip install numpy matplotlib jupyter
   ```

3. I start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. I open the notebook and run all cells from top to bottom.

## Verification

I executed every code cell in the final notebook. All included self-checks and assessment checks passed without execution errors, and the generated plots are embedded in the notebook.

## Author

Student ID: **2240001433**
