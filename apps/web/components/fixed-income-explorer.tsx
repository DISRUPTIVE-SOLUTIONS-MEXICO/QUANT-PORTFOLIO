"use client";

import { useMemo, useState } from "react";

import type { Row, Rows } from "@/lib/contracts";
import { finiteNumber } from "@/lib/contracts";

import { TimeSeriesChart } from "./charts";
import { DataTable } from "./data-table";
import { MetricStrip } from "./metric-strip";

function numeric(value: unknown): number {
  return finiteNumber(value);
}

function rate(value: Row[string]): string {
  const parsed = numeric(value);
  return Number.isFinite(parsed) ? `${parsed.toFixed(2)}%` : "Unavailable";
}

function basisPoints(value: Row[string]): string {
  const parsed = numeric(value);
  return Number.isFinite(parsed) ? `${parsed >= 0 ? "+" : ""}${parsed.toFixed(0)} bp` : "Unavailable";
}

function ratio(value: Row[string], digits = 2): string {
  const parsed = numeric(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "Unavailable";
}

export function FixedIncomeExplorer({
  metrics,
  factorHistory,
  scenarios,
}: {
  metrics: Rows;
  factorHistory: Rows;
  scenarios: Rows;
}) {
  const countries = useMemo(() => {
    const values = metrics.map((row) => String(row.Country ?? "")).filter(Boolean);
    return Array.from(new Set(values)).sort((left, right) => {
      if (left === "United States") return -1;
      if (right === "United States") return 1;
      return left.localeCompare(right);
    });
  }, [metrics]);
  const [selectedCountry, setSelectedCountry] = useState(countries[0] ?? "");
  const country = countries.includes(selectedCountry) ? selectedCountry : countries[0] ?? "";
  const selected = metrics.find((row) => String(row.Country) === country);
  const selectedHistory = useMemo(
    () =>
      factorHistory
        .filter((row) => String(row.Country) === country)
        .map((row) => ({
          Date: row.Date,
          "Curve level (%)": row.Level_Factor,
          "10Y-2Y slope (%)": row.Slope_10Y_2Y,
          "2Y-policy gap (%)": row.Policy_Gap_2Y,
        })),
    [country, factorHistory],
  );
  const selectedScenarios = useMemo(
    () => scenarios.filter((row) => String(row.Country) === country),
    [country, scenarios],
  );

  if (!selected || !metrics.length) {
    return (
      <div className="empty-state">
        <strong>Fixed-income intelligence unavailable</strong>
        <span>
          At least two source-verified sovereign maturities are required. The workbench does not interpolate a
          curve from a policy rate or a single observed tenor.
        </span>
      </div>
    );
  }

  const quality = String(selected.Curve_Quality ?? "Unavailable");
  const qualityScore = numeric(selected.Curve_Quality_Score);
  const slope = numeric(selected.Slope_10Y_2Y);
  return (
    <div className="fixed-income-workbench">
      <div className="fixed-income-toolbar">
        <div>
          <span>Sovereign market</span>
          <strong>{country}</strong>
          <small>
            {quality} evidence · native observation calendar · {String(selected.Rate_Source ?? "public source")}
          </small>
        </div>
        <label>
          Country
          <select value={country} onChange={(event) => setSelectedCountry(event.target.value)}>
            {countries.map((item) => (
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
            label: "Policy rate",
            value: rate(selected.Policy_Rate),
            detail: `As of ${String(selected.As_Of ?? "unavailable").slice(0, 10)}`,
          },
          {
            label: "2-year yield",
            value: rate(selected.Yield_2Y),
            detail: `Modified duration ${ratio(selected.Modified_Duration_2Y)}`,
          },
          {
            label: "10-year yield",
            value: rate(selected.Yield_10Y),
            detail: `Modified duration ${ratio(selected.Modified_Duration_10Y)}`,
          },
          {
            label: "10Y-2Y slope",
            value: rate(selected.Slope_10Y_2Y),
            detail: String(selected.Curve_State ?? "Unavailable"),
            tone: slope < 0 ? "warning" : "neutral",
          },
          {
            label: "Three-month slope change",
            value: basisPoints(selected.Slope_Change_3M_bp),
            detail: "Observed event-time change",
            tone: numeric(selected.Slope_Change_3M_bp) < -25 ? "warning" : "neutral",
          },
          {
            label: "Curve evidence quality",
            value: Number.isFinite(qualityScore) ? `${(qualityScore * 100).toFixed(0)}%` : "Unavailable",
            detail: `${String(selected.Sovereign_Tenor_Count ?? 0)} sovereign tenors · ${String(selected.History_Observations ?? 0)} states`,
            tone: quality === "High" ? "positive" : quality.startsWith("Insufficient") ? "warning" : "neutral",
          },
        ]}
      />
      <div className="chart-comparison fixed-income-comparison">
        <TimeSeriesChart
          rows={selectedHistory}
          title={`${country} observed term-structure factors`}
          emptyDetail="Historical factors require simultaneous source-verified 2-year and 10-year observations."
          referenceZero
        />
        <div className="fixed-income-readout">
          <h3>Curve and sensitivity state</h3>
          <dl>
            <div>
              <dt>Level factor</dt>
              <dd>{rate(selected.Level_Factor)}</dd>
            </div>
            <div>
              <dt>2Y-policy gap</dt>
              <dd>{rate(selected.Policy_Gap_2Y)}</dd>
            </div>
            <div>
              <dt>Level change, 3M</dt>
              <dd>{basisPoints(selected.Level_Change_3M_bp)}</dd>
            </div>
            <div>
              <dt>Observed rate-change volatility</dt>
              <dd>{basisPoints(selected.Observed_Rate_Change_Vol_bp)}</dd>
            </div>
            <div>
              <dt>Data staleness</dt>
              <dd>{String(selected.Stale_Days ?? "Unavailable")} days</dd>
            </div>
            <div>
              <dt>Observation mode</dt>
              <dd>Native event time</dd>
            </div>
          </dl>
        </div>
      </div>
      <DataTable
        rows={selectedScenarios}
        columns={[
          "Scenario",
          "Shock_2Y_bp",
          "Shock_10Y_bp",
          "Curve_Slope_Change_bp",
          "Approx_Price_Impact_2Y_pct",
          "Approx_Price_Impact_10Y_pct",
          "Equal_Notional_2Y_10Y_Impact_pct",
          "Interpretation",
        ]}
        emptyTitle="Duration and convexity scenarios unavailable"
        emptyDetail="Scenario sensitivities are withheld unless both observed 2-year and 10-year yields are present."
        maxRows={12}
      />
    </div>
  );
}
