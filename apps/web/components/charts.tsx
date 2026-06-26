"use client";

import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Rows } from "@/lib/contracts";
import { finiteNumber, normalizeSeriesLabel } from "@/lib/contracts";

const palette = ["#64c8ff", "#f2b84b", "#65d58b", "#b19cff", "#ff7b72", "#7ee7dc", "#c8d1dc"];

function MeasuredChart({ children }: { children: (width: number, height: number) => ReactNode }) {
  const stageRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;

    const measure = () => {
      const bounds = stage.getBoundingClientRect();
      const width = Math.max(0, Math.floor(bounds.width));
      const height = Math.max(0, Math.floor(bounds.height));
      setSize((current) => (current.width === width && current.height === height ? current : { width, height }));
    };

    measure();
    const frame = window.requestAnimationFrame(measure);
    window.addEventListener("resize", measure, { passive: true });
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => {
            measure();
          });
    observer?.observe(stage);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, []);

  return (
    <div className="chart-stage" ref={stageRef}>
      {size.width > 0 && size.height > 0 ? children(size.width, size.height) : null}
    </div>
  );
}

function compact(rows: Rows, max = 900): Rows {
  if (rows.length <= max) return rows;
  const step = Math.ceil(rows.length / max);
  return rows.filter((_, index) => index % step === 0 || index === rows.length - 1);
}

function numericKeys(rows: Rows, xKey: string): string[] {
  const counts = new Map<string, number>();
  const order: string[] = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (key === xKey) continue;
      if (!Number.isFinite(finiteNumber(row[key]))) continue;
      if (!counts.has(key)) order.push(key);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return order
    .filter((key) => (counts.get(key) ?? 0) >= Math.max(2, Math.floor(rows.length * 0.05)))
    .sort((left, right) => (counts.get(right) ?? 0) - (counts.get(left) ?? 0));
}

interface TimeSeriesChartProps {
  rows: Rows;
  title: string;
  xKey?: string;
  valueFormat?: "number" | "percent";
  emptyDetail: string;
  referenceZero?: boolean;
}

export function TimeSeriesChart({
  rows,
  title,
  xKey = "Date",
  valueFormat = "number",
  emptyDetail,
  referenceZero = false,
}: TimeSeriesChartProps) {
  const data = compact(rows);
  const keys = numericKeys(data, xKey).slice(0, 7);
  if (!data.length || !keys.length) {
    return (
      <section className="chart-panel">
        <h3>{title}</h3>
        <div className="empty-state">
          <strong>Series unavailable</strong>
          <span>{emptyDetail}</span>
        </div>
      </section>
    );
  }
  return (
    <section className="chart-panel">
      <h3>{title}</h3>
      <div className="chart-series-legend" aria-label={`${title} series`}>
        {keys.map((key, index) => (
          <span key={key}>
            <i style={{ backgroundColor: palette[index % palette.length] }} aria-hidden="true" />
            {normalizeSeriesLabel(key)}
          </span>
        ))}
      </div>
      <MeasuredChart>
        {(width, height) => (
          <LineChart width={width} height={height} data={data} margin={{ top: 16, right: 28, bottom: 48, left: 18 }}>
            <CartesianGrid stroke="#1f2a38" vertical={false} />
            <XAxis
              dataKey={xKey}
              interval="preserveStartEnd"
              minTickGap={64}
              stroke="#7f8b9d"
              tickMargin={12}
              tickFormatter={(value) => String(value).slice(0, 10)}
            />
            <YAxis
              stroke="#7f8b9d"
              width={68}
              tickMargin={8}
              tickFormatter={(value) =>
                valueFormat === "percent" ? `${(Number(value) * 100).toFixed(0)}%` : Number(value).toFixed(0)
              }
            />
            <Tooltip
              contentStyle={{ background: "#0c1119", border: "1px solid #263447", borderRadius: 4 }}
              labelStyle={{ color: "#dbe7f4" }}
              formatter={(value, name) => [
                valueFormat === "percent"
                  ? `${(Number(value) * 100).toFixed(2)}%`
                  : Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 }),
                normalizeSeriesLabel(String(name)),
              ]}
            />
            {referenceZero ? <ReferenceLine y={0} stroke="#8794a6" strokeDasharray="4 4" /> : null}
            {keys.map((key, index) => (
              <Line
                key={key}
                type="linear"
                dataKey={key}
                dot={false}
                stroke={palette[index % palette.length]}
                strokeWidth={key.toLowerCase().includes("benchmark") || key.includes("SPY") ? 1.7 : 2.2}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        )}
      </MeasuredChart>
    </section>
  );
}

export function YieldCurveChart({ rows, country = "United States" }: { rows: Rows; country?: string }) {
  const selected = rows.find((row) => String(row.Country) === country) ?? rows.find((row) => row.Yield_10Y !== null);
  const curve = selected
    ? [
        { Tenor: "Policy", Yield: finiteNumber(selected.Policy_Rate) },
        { Tenor: "2Y", Yield: finiteNumber(selected.Yield_2Y ?? selected.Yield_Short) },
        { Tenor: "10Y", Yield: finiteNumber(selected.Yield_10Y) },
      ].filter((point) => Number.isFinite(point.Yield))
    : [];
  if (curve.length < 2) {
    return (
      <section className="chart-panel">
        <h3>{country} sovereign curve</h3>
        <div className="empty-state">
          <strong>Insufficient real tenors</strong>
          <span>A curve requires at least two source-verified maturities. The last valid source remains visible in Data Quality.</span>
        </div>
      </section>
    );
  }
  return (
    <section className="chart-panel">
      <div className="chart-heading">
        <h3>{String(selected?.Country)} sovereign curve</h3>
        <span>{String(selected?.Rate_Source ?? "Public source")}</span>
      </div>
      <MeasuredChart>
        {(width, height) => (
          <LineChart width={width} height={height} data={curve} margin={{ top: 20, right: 34, bottom: 44, left: 14 }}>
            <CartesianGrid stroke="#1f2a38" vertical={false} />
            <XAxis dataKey="Tenor" stroke="#7f8b9d" tickMargin={12} />
            <YAxis stroke="#7f8b9d" unit="%" width={62} domain={["auto", "auto"]} tickMargin={8} />
            <Tooltip
              formatter={(value) => [`${Number(value).toFixed(2)}%`, "Yield"]}
              contentStyle={{ background: "#0c1119", border: "1px solid #263447", borderRadius: 4 }}
            />
            <Line dataKey="Yield" stroke="#64c8ff" strokeWidth={2.5} dot={{ r: 4 }} isAnimationActive={false} />
          </LineChart>
        )}
      </MeasuredChart>
    </section>
  );
}
