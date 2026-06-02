# How To Use Quant Portfolio-Kaizen

Quant Portfolio-Kaizen helps a user build a portfolio using public data, risk
constraints, causal backtesting and validation gates.

The basic workflow is:

1. Complete the risk profile questionnaire.
2. Select a ticker universe.
3. Choose a filter style: growth, value, quality, factor or custom.
4. Select a benchmark.
5. Run optimization.
6. Review the allocation status.
7. Read suitability and validation warnings before interpreting weights.

If the suitability gate blocks the portfolio, the allocation should not be
treated as a recommendation. The user should adjust horizon, drawdown tolerance,
capital, liquidity needs or constraints.

If the promotion gate rejects the result, the run is research-only. The model
needs stronger out-of-sample evidence.

