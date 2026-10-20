# My ARTI 402 Lab 7

**Student ID:** 2240001433

I completed Lab 7 using TensorFlow and Keras to classify the three-class spiral dataset. My notebook contains all completed exercises, the assessment, the actual execution outputs, and the requested figures.

## What I implemented

- I built a reusable network with two inputs, a configurable ReLU hidden layer, and three softmax outputs. With 128 hidden neurons, I verified that it has 771 parameters.
- I compiled it with Adam at a learning rate of 0.02 and sparse categorical cross-entropy for integer labels.
- I trained the initial model for 100 epochs with batches of 32.
- I added L2 regularization and an early-stopping callback with patience 25 and best-weight restoration.
- I used a repeating, full-batch `tf.data` pipeline for the four-way assessment, including dropout at 0.2 and assessment early stopping with patience 10.
- I saved and reloaded the early-stopped model and checked that its predictions matched.
- I plotted the baseline, L2, and dropout decision boundaries and the baseline learning curves.

## My assessment results

| Model | Training accuracy | Validation accuracy | Validation loss | Accuracy gap |
|---|---:|---:|---:|---:|
| baseline | 0.983 | 0.723 | 2.308 | +0.260 |
| L2 (5e-4) | 0.958 | 0.707 | 1.193 | +0.252 |
| dropout (0.2) | 0.892 | 0.673 | 1.856 | +0.218 |
| early stopping | 0.925 | 0.647 | 0.954 | +0.278 |

I found overfitting in the baseline. All three alternatives lowered validation loss in my run, and **early stopping** had the lowest validation loss. I included my interpretation below Q2 in the notebook. I did not use the test split to select a model.

## My files

- `arti402_Lab7_2240001433.ipynb`: my completed notebook with all code outputs and three embedded figures.
- `arti402_figures.py`: the supplied plotting helper, kept beside my notebook.
- `spiral_model.keras`: my saved Exercise 4 model, recreated when I run the notebook.
- `requirements.txt`: my tested package versions and Jupyter dependencies.
- `README.md`: this explanation of my work and how I rerun it.

I omitted the angle brackets around my ID because Windows does not allow `<` or `>` in filenames.

## How I run it again

I used Python 3.12.10, TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.5.2, and Matplotlib 3.11.1 on Windows. I use a short environment path to avoid Windows' path-length limit. From PowerShell in this folder, I can create an environment and register its Jupyter kernel:

```powershell
python -m venv "$env:LOCALAPPDATA\lab7-env"
& "$env:LOCALAPPDATA\lab7-env\Scripts\python.exe" -m pip install -r requirements.txt
& "$env:LOCALAPPDATA\lab7-env\Scripts\python.exe" -m ipykernel install --user --name arti402-lab7 --display-name "Python (ARTI402 Lab7)"
& "$env:LOCALAPPDATA\lab7-env\Scripts\python.exe" -m notebook .
```

I open the completed notebook, select **Python (ARTI402 Lab7)**, and choose **Restart Kernel and Run All Cells**. I keep the plotting helper in the same folder and run the cells from top to bottom. I do not need to download a dataset because the notebook generates the spiral data locally. Training time depends on my computer.

## My verification

I executed every code cell from a fresh kernel. All the original exercise and Q1 assertions passed, and the save/reload prediction assertion passed. The notebook has no recorded code-error outputs. My saved outputs include the model summaries, metrics, callback information, and all requested plots. Small numerical differences can occur with a different TensorFlow/Keras version or hardware.
