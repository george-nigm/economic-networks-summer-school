# Economic Networks Summer School: background notes

*Tutorial session, Thursday 25 June, Oxford.*

A short read before the tutorial. The point is to give you intuition, not formal definitions. Those you'll meet in the notebook.

## 1. Volatility

Volatility measures how much a price moves around, not where it's going. A market can be flat or it can be jumpy, and that jumpiness is what we call volatility. It's the working definition of risk.

The plain way to measure it is from closing prices: take daily returns and look at their spread. But close-to-close throws away everything that happened during the day. A price can swing wildly and come back, and you'd never know.

Parkinson volatility (the `vol` column in the data) fixes part of this by using the daily high and low instead. For a single day:

$$\sigma_P^2 = \frac{1}{4\ln 2}\left(\ln\frac{H}{L}\right)^2$$

where H is the day's high and L is the day's low. The intraday range carries more information than the close alone, so the estimate is tighter for the same amount of data (roughly five times as efficient).

One property matters above all: volatility clusters. Calm days follow calm days, violent days follow violent days. That's why we bother forecasting it at all.

## 2. What we use it for

We forecast volatility, not direction. Which way a price goes tomorrow is close to a coin flip. How *much* it will move is not, because of the clustering above. Volatility is also where the money and the risk actually live, so a good forecast feeds straight into decisions:

* **Trading volatility directly with options.** Expect volatility to rise, buy options (for example a straddle: a call and a put together) and you profit from a big move either way. Expect calm, sell options and collect the premium.
* **Position sizing.** Keep risk roughly constant. Volatility goes up, cut the position; volatility falls, add to it. Most systematic funds run on this.
* **Diversification.** Build a portfolio from weakly correlated assets so they don't all sink at once. The network shows you which assets are clustered and which stand alone.
* **Risk and contagion.** Volatility spills over: shake one market and related ones follow. The network tells you who the central, risk-spreading nodes are.
* **Pairs / stat-arb.** Two assets that usually move together drift apart, so buy the cheap one and sell the expensive one and bet they reconverge. This needs the correlations from the tutorial.

The dataset is daily futures across asset classes (equity index, FX, rates, commodities). For our purposes each symbol is just a time series; the interesting part is how they hang together.

## 3. Methods, and what each is for

Three steps, increasing in ambition.

**Correlation matrix.** Pivot the prices into a wide table and correlate every asset with every other. Reorder by similarity and blocks appear: equity indices cluster together, currencies elsewhere, and so on. This is how you *see* the structure before modelling it.

*How it's computed.* Pearson correlation between two assets X and Y is

$$\rho_{XY} = \frac{\text{cov}(X, Y)}{\sigma_X \, \sigma_Y}$$

their covariance divided by the product of their standard deviations. It runs from -1 (move exactly opposite) through 0 (unrelated) to +1 (move together). Do it for every pair and you have the matrix. We correlate the price series here because the block structure shows up clearly on prices; once you're comfortable, try the same on daily returns and see how the picture changes.

**Thresholded networks.** Keep a link between two assets only when their correlation is strong enough (above some threshold). Now you have a graph: assets are nodes, strong co-movements are edges. Slide the threshold and you trade density for clarity. At a high threshold only the market's skeleton survives. Build this on the training data, not the whole thing, or you're peeking at the future.

*How hierarchical clustering works.* Turn correlation into a distance (close assets, short distance). Start with every asset on its own, then repeatedly merge the two nearest groups, then the next nearest, and so on until everything is joined. The order of merges is a tree (a *dendrogram*): tight pairs join early and low, unrelated ones only at the top. Reorder the matrix by that tree and the blocks line up on the diagonal.

**Graph Convolutional Network (GCN).** A neural net that forecasts each asset's volatility using not just its own past but its neighbours' state in the network. Each layer lets a node mix in information from whoever it's connected to, which is exactly the spillover idea made concrete. Ours is a toy: the goal is to feel the mechanism, not to ship a trading system.

*How it works.* Each asset (node) starts with a feature vector, for example its recent volatility. One `GCNConv` layer is the Kipf–Welling rule:

$$H^{(l+1)} = \sigma\Big(\underbrace{\tilde{D}^{-1/2}\,\tilde{A}\,\tilde{D}^{-1/2}}_{\text{average over neighbours}}\;\underbrace{H^{(l)}\,W^{(l)}}_{\text{linear transform}}\Big)$$

Here $\tilde{A}=A+I$ is the adjacency matrix with self-loops added, $\tilde{D}$ is its degree matrix (so the left factor is just a normalised neighbour-average), $H^{(l)}$ are the node features at layer $l$, $W^{(l)}$ is the learned weight matrix, and $\sigma$ is a nonlinearity. In words: mix each node with its neighbours, apply a small learned transform, repeat. Stack two layers and a node sees its neighbours' neighbours too, so information spreads along the network.

*Worked example, all the way to a forecast.* Suppose the network links ES to NQ and to GC (gold), and today's volatilities are ES = 0.20, NQ = 0.30, GC = 0.40. We want tomorrow's ES volatility.

1. **Average over neighbours** (the left factor; including ES itself via the self-loop):
$$\tfrac{1}{3}(0.20 + 0.30 + 0.40) = 0.30$$
ES alone read 0.20, but blended with its agitated neighbours it becomes 0.30.

2. **Linear transform + nonlinearity.** Multiply by the learned weight, say $W = 0.9$, then apply ReLU:
$$\sigma(0.30 \times 0.9) = 0.27$$
That is ES's hidden feature after one layer. (A second layer would repeat steps 1–2, pulling in neighbours-of-neighbours.)

3. **Readout.** A final linear layer turns the hidden feature into the prediction. With weight 1.0 and no bias:
$$\widehat{\text{ES}}_\text{tomorrow} = 0.27$$

So ES read 0.20 today, yet the model forecasts about 0.27 tomorrow, pulled up because its neighbours are running hot. That upward nudge is the spillover the network exists to capture. Training just tweaks $W$ and the readout weights so these forecasts land as close as possible to what volatility actually does next.