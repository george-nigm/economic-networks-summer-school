# Economic Networks Summer School

Materials for the tutorial session, **Thursday 25 June, Oxford**.

We build networks from multivariate daily futures data (equity index, FX, rates, commodities) and use them to forecast volatility. Work through the files in order.

## Contents

| File | What it is |
|------|------------|
| `01_background.md` | Short primer: volatility, the forecasting problem, and the methods. Read this first. |
| `02_tutorial.ipynb` | The tutorial: correlation matrix, thresholded networks, then a toy GCN forecast. |
| `merged_multivariate.csv` | Daily OHLC, volume and Parkinson volatility per asset, from 2008. |

## Setup

Download or clone the folder, open it in your editor (VS Code works well), then in the terminal create an environment and install the dependencies.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then launch the notebook:

```bash
jupyter lab 02_tutorial.ipynb
```

It expects `merged_multivariate.csv` in the same folder. If you'd rather not install anything, open `02_tutorial.ipynb` in Colab instead.

## Exercises

1. Pivot to a wide price table, compute the correlation matrix, look for hierarchical structure.
2. Threshold the absolute correlations to build and visualise networks.
3. **Challenge:** using a network from (2) on the training set, build a small Graph Convolutional Network to forecast daily Parkinson volatility.

## Questions

If you have any questions, get in touch with us:

- George Nigmatulin, george.nigmatulin@eng.ox.ac.uk
- Yaxuan Kong, yaxuan.kong@eng.ox.ac.uk
