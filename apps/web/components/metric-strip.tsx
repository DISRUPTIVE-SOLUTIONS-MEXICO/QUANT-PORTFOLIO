import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

export interface MetricItem {
  label: string;
  value: string;
  detail?: string;
  tone?: "positive" | "negative" | "neutral" | "warning";
}

export function MetricStrip({ metrics }: { metrics: MetricItem[] }) {
  return (
    <div className="metric-strip">
      {metrics.map((metric) => {
        const Icon =
          metric.tone === "positive" ? ArrowUpRight : metric.tone === "negative" ? ArrowDownRight : Minus;
        const missing = metric.value === "Unavailable" || metric.value === "";
        const displayValue = missing ? "n/a" : metric.value;
        return (
          <section className={`metric-readout tone-${metric.tone ?? "neutral"} ${missing ? "is-missing" : ""}`} key={metric.label}>
            <span>{metric.label}</span>
            <strong>{displayValue}</strong>
            <small>
              <Icon aria-hidden="true" size={14} />
              {metric.detail ?? "No comparison available"}
            </small>
          </section>
        );
      })}
    </div>
  );
}