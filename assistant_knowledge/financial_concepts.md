# Financial Concepts

Sortino ratio measures return per unit of downside volatility. It penalizes
negative returns more directly than Sharpe.

CVaR estimates the average loss conditional on being in the tail beyond VaR.
It is a left-tail risk metric and is more conservative than volatility.

Drawdown measures loss from a prior peak:

```text
drawdown = price / running_peak - 1
```

GARCH models conditional volatility clustering. Student-t GARCH is used when
returns have heavy tails.

PELT detects structural changes in mean or variance. It is used to identify
regime changes in portfolio behavior.

PBO, Deflated Sortino, White Reality Check and Hansen SPA are validation tools
that reduce the probability of selecting a strategy merely because it overfit
historical data.

PIT confidence estimates whether a fundamental ratio is close to point-in-time
evidence or merely a public-data approximation.

