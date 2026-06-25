# Economic Networks Summer School — background notes

*Tutorial session, Thursday 25 June, Oxford.*

A short read before the tutorial. The point is to give you intuition, not formal definitions — those you'll meet in the notebook.

## 1. Volatility

Volatility measures how much a price moves around, not where it's going. A market can be flat or it can be jumpy, and that jumpiness is what we call volatility. It's the working definition of risk.

The plain way to measure it is from closing prices: take daily returns and look at their spread. But close-to-close throws away everything that happened during the day — a price can swing wildly and come back, and you'd never know.

Parkinson volatility (the `vol` column in the data) fixes part of this by using the daily high and low instead. The intraday range carries more information than the close alone, so the estimate is tighter for the same amount of data. It has blind spots — it misses overnight gaps and assumes no drift — but for daily futures it does the job.

One property matters above all: volatility clusters. Calm days follow calm days, violent days follow violent days. That's why we bother forecasting it at all.

## 2. What we're actually trying to do

Note that we forecast volatility, not direction. Which way a price goes tomorrow is close to a coin flip. How *much* it will move is not — the clustering above makes it predictable enough to be useful.

Why anyone cares:

- It's the price tag on risk. Position sizing, stop levels, option prices all run on it.
- It spills over. When one market gets shaken, related ones tend to follow. So a forecast for one asset can borrow strength from its neighbours — which is the whole reason we drag networks into this.

The dataset is daily futures across asset classes (equity index, FX, rates, commodities). For our purposes each symbol is just a time series; the interesting part is how they hang together.

## 3. Methods, and what each is for

Three steps, increasing in ambition.

**Correlation matrix.** Pivot the prices into a wide table and correlate every asset with every other. Reorder by similarity and blocks appear — equity indices cluster together, currencies elsewhere, and so on. This is how you *see* the structure before modelling it.

**Thresholded networks.** Keep a link between two assets only when their correlation is strong enough (above some threshold). Now you have a graph: assets are nodes, strong co-movements are edges. Slide the threshold and you trade density for clarity — at a high threshold only the market's skeleton survives. Build this on the training data, not the whole thing, or you're peeking at the future.

**Graph Convolutional Network.** A neural net that forecasts each asset's volatility using not just its own past but its neighbours' state in the network. Each layer lets a node mix in information from whoever it's connected to, which is exactly the spillover idea made concrete. Ours is a toy — the goal is to feel the mechanism, not to ship a trading system.
