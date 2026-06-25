# Economic Networks Summer School: background notes

*Tutorial session, Thursday 25 June, Oxford.*

A short read before the tutorial. The point is to give you intuition, not formal definitions. Those you'll meet in the notebook.

## 1. Volatility

Volatility measures how much a price moves around, not where it's going. A market can be flat or it can be jumpy, and that jumpiness is what we call volatility. It's the working definition of risk.

The plain way to measure it is from closing prices: take daily returns and look at their spread. But close-to-close throws away everything that happened during the day. A price can swing wildly and come back, and you'd never know.

Parkinson volatility (the `vol` column in the data) fixes part of this by using the daily high and low instead. For a single day:

$$\sigma_P^2 = \frac{1}{4\ln 2}\left(\ln\frac{H}{L}\right)^2$$

where H is the day's high and L is the day's low. The intraday range carries more information than the close alone, so the estimate is tighter for the same amount of data (roughly five times as efficient). It has blind spots: it misses overnight gaps and assumes no drift. For daily futures it does the job.

One property matters above all: volatility clusters. Calm days follow calm days, violent days follow violent days. That's why we bother forecasting it at all.

## 2. What we use it for

We forecast volatility, not direction. Which way a price goes tomorrow is close to a coin flip. How *much* it will move is not, because of the clustering above. Volatility is also where the money and the risk actually live, so a good forecast feeds straight into decisions:

* **Position sizing.** Keep risk roughly constant. Volatility goes up, cut the position; volatility falls, add to it. Most systematic funds run on this.
* **Trading volatility directly with options.** Expect volatility to rise, buy options (for example a straddle: a call and a put together) and you profit from a big move either way. Expect calm, sell options and collect the premium.
* **Pairs / stat-arb.** Two assets that usually move together drift apart, so buy the cheap one and sell the expensive one and bet they reconverge. This needs the correlations from the tutorial.
* **Diversification.** Build a portfolio from weakly correlated assets so they don't all sink at once. The network shows you which assets are clustered and which stand alone.
* **Risk and contagion.** Volatility spills over: shake one market and related ones follow. The network tells you who the central, risk-spreading nodes are.

The dataset is daily futures across asset classes (equity index, FX, rates, commodities). For our purposes each symbol is just a time series; the interesting part is how they hang together.

## 3. Methods, and what each is for

Three steps, increasing in ambition.

**Correlation matrix.** Pivot the prices into a wide table and correlate every asset with every other. Reorder by similarity and blocks appear: equity indices cluster together, currencies elsewhere, and so on. This is how you *see* the structure before modelling it.

*How it's computed.* Pearson correlation between two assets X and Y is

$$\rho_{XY} = \frac{\text{cov}(X, Y)}{\sigma_X \, \sigma_Y}$$

their covariance divided by the product of their standard deviations. It runs from -1 (move exactly opposite) through 0 (unrelated) to +1 (move together). Do it for every pair and you have the matrix. In practice correlate the daily *returns*, not the raw prices, so trends don't fake a link.

**Thresholded networks.** Keep a link between two assets only when their correlation is strong enough (above some threshold). Now you have a graph: assets are nodes, strong co-movements are edges. Slide the threshold and you trade density for clarity. At a high threshold only the market's skeleton survives. Build this on the training data, not the whole thing, or you're peeking at the future.

*How hierarchical clustering works.* Turn correlation into a distance (close assets, short distance). Start with every asset on its own, then repeatedly merge the two nearest groups, then the next nearest, and so on until everything is joined. The order of merges is a tree (a *dendrogram*): tight pairs join early and low, unrelated ones only at the top. Reorder the matrix by that tree and the blocks line up on the diagonal.

**Graph Convolutional Network (GCN).** A neural net that forecasts each asset's volatility using not just its own past but its neighbours' state in the network. Each layer lets a node mix in information from whoever it's connected to, which is exactly the spillover idea made concrete. Ours is a toy: the goal is to feel the mechanism, not to ship a trading system.

*How it works.* Each asset (node) starts with a feature vector, for example its recent volatility. One GCN layer updates every node by averaging its own features with its neighbours', then applying a small learned transform. Stack two layers and a node sees its neighbours' neighbours too, so information spreads along the network. Train it by comparing the predicted next-day volatility against what actually happened and nudging the weights to shrink that error.
