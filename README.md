# Economic Networks Summer School

Materials for the tutorial session — **Thursday 25 June, Oxford**.

We build networks from multivariate daily futures data (equity index, FX, rates, commodities) and use them to forecast volatility. Read the background note first, then work through the notebook.

## Contents

| File | What it is |
|------|------------|
| `background.md` | Short primer — volatility, the forecasting problem, and the methods. Read before the session. |
| `tutorial.ipynb` | The tutorial: correlation matrix → thresholded networks → a toy GCN forecast. |
| `tutorial.pdf` | Rendered copy of the notebook, for reading offline. |
| `merged_multivariate.csv` | Daily OHLC, volume and Parkinson volatility per asset, from 2008. |

## Running it

The notebook expects `merged_multivariate.csv` in the same folder. You'll need:

```
pandas  numpy  matplotlib  networkx  torch  torch_geometric
```

Either run locally or open `tutorial.ipynb` in Colab.

## Exercises

1. Pivot to a wide price table, compute the correlation matrix, look for hierarchical structure.
2. Threshold the absolute correlations to build and visualise networks.
3. **Challenge:** using a network from (2) on the training set, build a small Graph Convolutional Network to forecast daily Parkinson volatility.
