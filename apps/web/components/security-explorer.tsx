"use client";

import { useMemo, useState } from "react";

import type { Row, Rows } from "@/lib/contracts";
import { finiteNumber } from "@/lib/contracts";

import { DataTable } from "./data-table";
import { MetricStrip } from "./metric-strip";
import { TimeSeriesChart } from "./charts";

function numeric(value: unknown): number {
  return finiteNumber(value);
}

function percent(value: Row[string]): string {
  const parsed = numeric(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(1)}%` : "Unavailable";
}

function ratio(value: Row[string]): string {
  const parsed = numeric(value);
  return Number.isFinite(parsed) ? parsed.toFixed(2) : "Unavailable";
}

function price(value: Row[string]): string {
  const parsed = numeric(value);
  return Number.isFinite(parsed)
    ? parsed.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : "Unavailable";
}

function compactDollars(value: Row[string]): string {
  const parsed = numeric(value);
  if (!Number.isFinite(parsed)) return "Unavailable";
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 1,
  }).format(parsed);
}

export function SecurityExplorer({
  metrics,
  priceHistory,
  consensus,
  benchmark,
}: {
  metrics: Rows;
  priceHistory: Rows;
  consensus: Rows;
  benchmark: string;
}) {
  const tickers = useMemo(
    () =>
      metrics
        .map((row) => String(row.Ticker ?? ""))
        .filter((ticker) => ticker && ticker !== benchmark),
    [benchmark, metrics],
  );
  const [selectedTicker, setSelectedTicker] = useState(tickers[0] ?? benchmark);
  const ticker = tickers.includes(selectedTicker) ? selectedTicker : tickers[0] ?? benchmark;
  const security = metrics.find((row) => String(row.Ticker) === ticker);
  const consensusRow = consensus.find((row) => String(row.Ticker) === ticker);
  const chartRows = useMemo(
    () =>
      priceHistory
        .map((row) => ({
          Date: row.Date,
          [`${ticker} observed price`]: row[ticker] ?? null,
          ...(ticker !== benchmark ? { [`${benchmark} observed price`]: row[benchmark] ?? null } : {}),
        }))
        .filter((row) => Number.isFinite(finiteNumber(row[`${ticker} observed price`]))),
    [benchmark, priceHistory, ticker],
  );

  if (!security || !metrics.length) {
    return (
      <div className="empty-state">
        <strong>Security intelligence unavailable</strong>
        <span>
          A source-aligned price history with at least 252 observations is required. The workbench never infers
          missing betas, liquidity or tail diagnostics.
        </span>
      </div>
    );
  }

  const residualMomentum = numeric(security.Residual_Momentum_126D);
  const tailBeta = numeric(security.Tail_Beta_252D);
  const selectionBreadth = numeric(security.Strategy_Selection_Breadth);
  return (
    <div className="security-workbench">
      <div className="security-toolbar">
        <div>
          <span>Instrument</span>
          <strong>{ticker}</strong>
          <small>
            Live snapshot · ξ {benchmark} · {String(security.Trend_State ?? "state unavailable")}
          </small>
        </div>
        <label>
          Security
          <select value={ticker} onChange={(event) => setSelectedTicker(event.target.value)}>
            {tickers.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>
      <MetricStrip
        metrics={[
          {
            label: "Observed price",
            value: price(security.Observed_Price),
            detail: `As of ${String(security.As_Of ?? "unavailable").slice(0, 10)}`,
          },
          {
            label: "12-month return",
            value: percent(security.Return_252D),
            detail: `3-month ${percent(security.Return_63D)}`,
            tone: numeric(security.Return_252D) >= 0 ? "positive" : "negative",
          },
          {
            label: "Residual momentum",
            value: percent(security.Residual_Momentum_126D),
            detail: "Market-model residual, 126 observations",
            tone: residualMomentum >= 0 ? "positive" : "warning",
          },
          {
            label: "Asymmetric beta",
            value: `${ratio(security.Upside_Beta_252D)} / ${ratio(security.Downside_Beta_252D)}`,
            detail: "Upside β / downside β",
            tone:
              numeric(security.Upside_Beta_252D) > numeric(security.Downside_Beta_252D)
                ? "positive"
                : "warning",
          },
          {
            label: "Tail beta",
            value: ratio(security.Tail_Beta_252D),
            detail: "Conditional on ξ below its 10% quantile",
            tone: tailBeta < 1 ? "positive" : "warning",
          },
          {
            label: "Strategy breadth",
            value: percent(security.Strategy_Selection_Breadth),
            detail: `${String(security.Strategies_Selected ?? 0)} active strategy families`,
            tone: selectionBreadth >= 0.5 ? "positive" : "neutral",
          },
        ]}
      />
      <div className="chart-comparison security-comparison">
        <TimeSeriesChart
          rows={chartRows}
          title={`${ticker} observed price versus benchmark ξ`}
          emptyDetail="The selected security does not have a benchmark-aligned observed price path."
        />
        <div className="security-risk-readout">
          <h3>Risk and implementability</h3>
          <dl>
            <div>
              <dt>Annualized volatility, 63D</dt>
              <dd>{percent(security.Annualized_Vol_63D)}</dd>
            </div>
            <div>
              <dt>Downside deviation, 63D</dt>
              <dd>{percent(security.Downside_Deviation_63D)}</dd>
            </div>
            <div>
              <dt>Maximum drawdown, 252D</dt>
              <dd>{percent(security.Max_Drawdown_252D)}</dd>
            </div>
            <div>
              <dt>Daily CVaR 95%</dt>
              <dd>{percent(security.CVaR_95_Daily)}</dd>
            </div>
            <div>
              <dt>Average dollar volume, 20D</dt>
              <dd>{compactDollars(security.ADV_USD_20D)}</dd>
            </div>
            <div>
              <dt>Observations</dt>
              <dd>{String(security.Observations ?? "Unavailable")}</dd>
            </div>
          </dl>
        </div>
      </div>
      <DataTable
        rows={[{ ...security, ...consensusRow }]}
        columns={[
          "Ticker",
          "Trend_State",
          "Return_21D",
          "Return_63D",
          "Return_126D",
          "Return_252D",
          "Beta_to_Xi_252D",
          "Correlation_to_Xi_252D",
          "Upside_Beta_252D",
          "Downside_Beta_252D",
          "Tail_Beta_252D",
          "Idiosyncratic_Vol_252D",
          "Current_Drawdown",
          "Recovery_Days_From_Trough",
          "ADV_USD_20D",
          "Amihud_ILLIQ_63D",
          "Consensus_Rank_0_1",
          "Strategies_Selected",
        ]}
        emptyTitle="Instrument diagnostics unavailable"
        emptyDetail="The selected instrument has no admissible causal diagnostics."
      />
    </div>
  );
}
