import { AlertTriangle, Ban, ShieldCheck } from "lucide-react";
import Link from "next/link";

import type { DashboardBundle, Row } from "@/lib/contracts";
import { metricMap, rows, section } from "@/lib/contracts";
import { workspaces } from "@/lib/navigation";

import { DataTable } from "./data-table";
import { MathPanel } from "./math-panel";
import { MetricStrip, type MetricItem } from "./metric-strip";
import { PageHeader } from "./page-header";
import { SectionHeading } from "./section-heading";
import { TimeSeriesChart } from "./charts";

function percent(value: number | undefined): string {
  return Number.isFinite(value) ? `${(Number(value) * 100).toFixed(1)}%` : "Unavailable";
}

function number(value: number | undefined): string {
  return Number.isFinite(value) ? Number(value).toFixed(3) : "Unavailable";
}

function benchmark(payload: DashboardBundle["merged"]): string {
  const registry = rows(section(payload, "research").model_registry);
  const context = rows(section(payload, "status").market_context);
  return String(
    registry[0]?.benchmark_ticker ??
      registry[0]?.Benchmark ??
      context[0]?.Benchmark ??
      context[0]?.Benchmark_Ticker ??
      "ξ",
  );
}

function promotionRow(payload: DashboardBundle["merged"]): Row | undefined {
  return rows(section(payload, "status").promotion)[0];
}

function completenessReadouts(payload: DashboardBundle["merged"]): MetricItem[] {
  const matrix = rows(section(payload, "status").capability_completeness);
  if (!matrix.length) {
    return [
      {
        label: "Artifact completeness",
        value: "Unavailable",
        detail: "No capability matrix in active payload",
        tone: "warning",
      },
    ];
  }
  const scores = matrix
    .map((row) => Number(row.Completeness))
    .filter((value) => Number.isFinite(value));
  const average = scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : Number.NaN;
  const complete = matrix.filter((row) => String(row.Status ?? "").toLowerCase() === "complete").length;
  const missing = matrix.filter((row) => String(row.Status ?? "").toLowerCase() === "missing").length;
  return [
    {
      label: "Artifact completeness",
      value: Number.isFinite(average) ? `${(average * 100).toFixed(0)}%` : "Unavailable",
      detail: `${complete}/${matrix.length} modules complete`,
      tone: Number.isFinite(average) && average >= 0.8 ? "positive" : "warning",
    },
    {
      label: "Missing modules",
      value: String(missing),
      detail: "Visible in Data Quality",
      tone: missing === 0 ? "positive" : "warning",
    },
  ];
}

type DecisionState = "approved" | "research" | "blocked";

function truthyPass(value: unknown): boolean {
  if (typeof value === "boolean") return value;
  return ["true", "pass", "passed", "approved", "promoted"].includes(String(value ?? "").trim().toLowerCase());
}

function decisionGate(payload: DashboardBundle["merged"]): {
  state: DecisionState;
  label: string;
  detail: string;
  failure: string;
} {
  const status = section(payload, "status");
  const promotion = rows(status.promotion)[0];
  const suitability = rows(status.suitability)[0];
  const breaches = rows(status.suitability_breaches);
  const tests = rows(status.promotion_tests);
  const promotionStatus = String(promotion?.Promotion_Status ?? promotion?.promotion_status ?? "").trim().toLowerCase();
  const suitabilityStatus = String(
    suitability?.Gate_Status ?? suitability?.Suitability_Status ?? suitability?.status ?? "",
  )
    .trim()
    .toLowerCase();
  const failedTest = tests.find((row) => "Pass" in row && !truthyPass(row.Pass));
  const failedLabel = String(failedTest?.Test ?? failedTest?.Metric ?? failedTest?.Name ?? "");

  if (breaches.length > 0 || ["blocked", "rejected", "failed", "breach"].some((term) => suitabilityStatus.includes(term))) {
    return {
      state: "blocked",
      label: "BLOCKED",
      detail: "Suitability or mandate constraints prohibit allocation and paper execution.",
      failure: breaches.length ? `${breaches.length} suitability breach${breaches.length === 1 ? "" : "es"}` : suitabilityStatus,
    };
  }
  const promotionApproved = ["promoted", "approved"].includes(promotionStatus);
  const testsComplete = tests.length > 0 && tests.every((row) => !("Pass" in row) || truthyPass(row.Pass));
  if (promotionApproved && testsComplete) {
    return {
      state: "approved",
      label: "APPROVED",
      detail: "Pre-registered OOS, downside and governance gates permit paper execution.",
      failure: "No active gate failure",
    };
  }
  return {
    state: "research",
    label: "RESEARCH-ONLY",
    detail: "Evidence remains visible, but the strategy cannot progress to an executable recommendation.",
    failure: failedLabel ? `First failed gate: ${failedLabel}` : "Promotion evidence incomplete",
  };
}

function decisionBriefing(payload: DashboardBundle["merged"], xi: string): { title: string; value: string; detail: string; tone: string }[] {
  const metrics = metricMap(payload);
  const status = section(payload, "status");
  const completeness = rows(status.capability_completeness);
  const activeReturn =
    metrics.has("Annualized_Return") && metrics.has("Benchmark_Annualized_Return")
      ? Number(metrics.get("Annualized_Return")) - Number(metrics.get("Benchmark_Annualized_Return"))
      : Number.NaN;
  const cvar = Number(metrics.get("CVaR_95_Daily"));
  const benchmarkCvar = Number(metrics.get("Benchmark_CVaR_95_Daily"));
  const hasCompletenessMatrix = completeness.length > 0;
  const missingModules = completeness
    .filter((row) => String(row.Status ?? "").toLowerCase() !== "complete")
    .map((row) => String(row.Module ?? ""))
    .filter(Boolean);
  return [
    {
      title: "Benchmark-relative edge",
      value: Number.isFinite(activeReturn) ? percent(activeReturn) : "Unavailable",
      detail: Number.isFinite(activeReturn)
        ? activeReturn >= 0
          ? `Portfolio is ahead of ${xi}; verify promotion gates before action.`
          : `Portfolio trails ${xi}; inspect XCDR growth sleeve and downside budget.`
        : "Active return is absent from the full evidence artifact.",
      tone: Number.isFinite(activeReturn) && activeReturn >= 0 ? "positive" : "warning",
    },
    {
      title: "Downside integrity",
      value: Number.isFinite(cvar) ? percent(cvar) : "Unavailable",
      detail:
        Number.isFinite(cvar) && Number.isFinite(benchmarkCvar)
          ? cvar <= benchmarkCvar
            ? "Tail loss is inside the benchmark envelope."
            : `Tail loss exceeds ${xi}; no promotion without CVaR repair.`
          : "CVaR comparison is missing; keep the strategy research-only.",
      tone: Number.isFinite(cvar) && Number.isFinite(benchmarkCvar) && cvar <= benchmarkCvar ? "positive" : "warning",
    },
    {
      title: "Evidence coverage",
      value: !hasCompletenessMatrix ? "Unavailable" : missingModules.length ? `${missingModules.length} gaps` : "Complete",
      detail: !hasCompletenessMatrix
        ? "The active artifact has no capability matrix; treat this view as incomplete until Data Quality confirms module coverage."
        : missingModules.length
        ? `Open Data Quality: ${missingModules.slice(0, 3).join(", ")}${missingModules.length > 3 ? "..." : ""}.`
        : "The active artifact contains all required institutional modules.",
      tone: !hasCompletenessMatrix || missingModules.length ? "warning" : "positive",
    },
  ];
}

const CAPABILITY_WORKSPACE: Record<string, string> = {
  "Market Intelligence": "Market Intelligence",
  "Rates & Fixed Income": "Rates & Fixed Income",
  "Equity Fundamentals": "Equity Research",
  "Benchmark xi": "Validation & Governance",
  "XCDR Research": "XCDR Research",
  "Portfolio Construction": "Portfolio Construction",
  "Risk Laboratory": "Risk Laboratory",
  "Validation & Governance": "Validation & Governance",
  "Data Quality": "Data Quality",
};

function workspaceHref(module: string): string {
  const label = CAPABILITY_WORKSPACE[module] ?? "Data Quality";
  const workspace = workspaces.find((item) => item.label === label);
  return workspace?.slug ? `/${workspace.slug}` : "/";
}

function evidencePercent(row: Row): string {
  const score = Number(row.Completeness);
  return Number.isFinite(score) ? `${Math.round(score * 100)}%` : "n/a";
}

function evidenceTone(row: Row): string {
  const status = String(row.Status ?? "").toLowerCase();
  if (status === "complete") return "complete";
  if (status === "missing") return "missing";
  return "partial";
}

const DECISION_FLOW = [
  { label: "Public data", detail: "Yahoo, SEC, FRED, rates, options, news and source freshness." },
  { label: "PIT quality", detail: "Availability dates, cache state, null coverage and public-data confidence." },
  { label: "Signals", detail: "Sector-normalized fundamentals, residual alpha, RMT, entropy and tail state." },
  { label: "Benchmark ξ", detail: "Mandate-compatible reference selected before optimization." },
  { label: "XCDR policy", detail: "Upside capture under downside, CVaR, drawdown and uncertainty budgets." },
  { label: "Governance", detail: "WRC, SPA, PBO, ICIR, suitability and paper-execution controls." },
];

export function CommandCenter({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.merged;
  const metrics = metricMap(payload);
  const xi = benchmark(payload);
  const activeReturn =
    metrics.has("Annualized_Return") && metrics.has("Benchmark_Annualized_Return")
      ? Number(metrics.get("Annualized_Return")) - Number(metrics.get("Benchmark_Annualized_Return"))
      : undefined;
  const readouts: MetricItem[] = [
    {
      label: "Annualized return",
      value: percent(metrics.get("Annualized_Return")),
      detail: `${xi} ${percent(metrics.get("Benchmark_Annualized_Return"))}`,
      tone: activeReturn === undefined ? "neutral" : activeReturn >= 0 ? "positive" : "negative",
    },
    {
      label: "Active return vs ξ",
      value: percent(activeReturn),
      detail: "OOS benchmark-relative result",
      tone: activeReturn === undefined ? "neutral" : activeReturn >= 0 ? "positive" : "negative",
    },
    {
      label: "Annualized volatility",
      value: percent(metrics.get("Annualized_Vol")),
      detail: `${xi} ${percent(metrics.get("Benchmark_Annualized_Vol"))}`,
      tone:
        Number(metrics.get("Annualized_Vol")) <= Number(metrics.get("Benchmark_Annualized_Vol"))
          ? "positive"
          : "warning",
    },
    {
      label: "Maximum drawdown",
      value: percent(metrics.get("Max_Drawdown")),
      detail: `${xi} ${percent(metrics.get("Benchmark_Max_Drawdown"))}`,
      tone:
        Number(metrics.get("Max_Drawdown")) >= Number(metrics.get("Benchmark_Max_Drawdown"))
          ? "positive"
          : "warning",
    },
    {
      label: "XCDR v3",
      value: number(metrics.get("XCDR_v3")),
      detail: "Return under downside uncertainty",
      tone: Number(metrics.get("XCDR_v3")) > 0 ? "positive" : "neutral",
    },
  ];
  const charts = section(payload, "charts");
  const status = section(payload, "status");
  const promotion = promotionRow(payload);
  const gate = decisionGate(payload);
  const briefing = decisionBriefing(payload, xi);
  const capabilityRows = rows(status.capability_completeness);
  const GateIcon = gate.state === "approved" ? ShieldCheck : gate.state === "blocked" ? Ban : AlertTriangle;

  return (
    <div className="workspace-page">
      <PageHeader
        eyebrow="Portfolio decision system"
        title="Command Center"
        description="XCDR research evidence, benchmark-relative risk and the freshest validated market overlay in one auditable view."
        scope={bundle.full ? "OOS research + daily overlay" : "Daily snapshot only"}
      />
      {bundle.notices.map((notice) => (
        <div className="notice" role="status" key={notice}>
          {notice}
        </div>
      ))}
      <section className={`decision-gate decision-${gate.state}`} aria-label="Allocation decision gate" role="status">
        <div className="decision-gate-state">
          <GateIcon aria-hidden="true" />
          <div>
            <span>Allocation decision</span>
            <strong>{gate.label}</strong>
          </div>
        </div>
        <p>{gate.detail}</p>
        <div className="decision-gate-evidence">
          <span>Binding evidence</span>
          <strong>{gate.failure}</strong>
        </div>
      </section>
      <MetricStrip metrics={readouts} />
      <section className="decision-briefing" aria-label="Decision briefing">
        {briefing.map((item) => (
          <article className={`briefing-card tone-${item.tone}`} key={item.title}>
            <span>{item.title}</span>
            <strong>{item.value}</strong>
            <p>{item.detail}</p>
          </article>
        ))}
      </section>
      <MetricStrip metrics={completenessReadouts(payload)} />

      <section className="terminal-map" aria-label="Institutional research terminal map">
        <div className="terminal-map-head">
          <div>
            <span>Analytical surface map</span>
            <strong>Complete terminal, not a thin snapshot</strong>
          </div>
          <p>
            Each tile is a preserved capability. Missing evidence is routed to Data Quality instead of being hidden or replaced by a cosmetic placeholder.
          </p>
        </div>
        <div className="terminal-map-grid">
          {capabilityRows.map((row) => {
            const capability = String(row.Module ?? "Module");
            const tone = evidenceTone(row);
            const missing = String(row.Missing_Evidence ?? "").trim();
            return (
              <Link className={`terminal-map-card terminal-map-${tone}`} href={workspaceHref(capability)} key={capability}>
                <span>{String(row.Freshness_Requirement ?? "publication")}</span>
                <strong>{capability}</strong>
                <p>{String(row.Description ?? "Institutional evidence module.")}</p>
                <div>
                  <b>{evidencePercent(row)}</b>
                  <em>{String(row.Status ?? "unknown")}</em>
                </div>
                <small>{missing ? `Missing: ${missing}` : "Evidence ready for analytical drill-down."}</small>
              </Link>
            );
          })}
        </div>
      </section>

      <section className="decision-flow" aria-label="Causal decision workflow">
        {DECISION_FLOW.map((step, index) => (
          <article key={step.label}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{step.label}</strong>
            <p>{step.detail}</p>
          </article>
        ))}
      </section>

      <SectionHeading
        title={`XCDR portfolio versus optimal benchmark ${xi}`}
        description="Observed benchmark prices and the causal OOS portfolio price path are shown on a common currency scale. Drawdown is computed independently from each daily price path."
        evidence={bundle.full?.createdAt ? `Full evidence ${new Date(bundle.full.createdAt).toLocaleDateString("en-US")}` : "Full evidence unavailable"}
      />
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={rows(charts.price_paths)}
          title="Causal OOS price paths"
          emptyDetail="A full research artifact with at least three years of daily prices is required."
        />
        <TimeSeriesChart
          rows={rows(charts.drawdowns)}
          title="Daily drawdown paths"
          valueFormat="percent"
          referenceZero
          emptyDetail="Drawdown requires a validated daily price path and running peak."
        />
      </div>

      <MathPanel
        title="Benchmark-relative XCDR construction"
        formula={String.raw`\max_{w\in\Delta_N}\;\frac{AR(w,\xi)+\lambda_U UC(w,\xi)+\lambda_C(\beta_p^+-\beta_p^-)}{\epsilon+D_-(w)+\lambda_{\mathrm{CVaR}}\,\mathrm{CVaR}(w)+\lambda_{\mathrm{DD}}\,\mathrm{DD}(w)+\lambda_U\,U(w)+\lambda_T\,\mathrm{TO}(w)}`}
      >
        <p>
          The benchmark <strong>ξ</strong> is selected before allocation from mandate fit, factor overlap,
          country, sector, beta and tracking-error compatibility. The broader Ω set is a stress universe,
          not a target that can be selected after observing test returns.
        </p>
      </MathPanel>

      <div className="split-band">
        <section>
          <SectionHeading
            title="Promotion evidence"
            description="A strategy remains research-only unless the pre-registered OOS gates pass."
          />
          <DataTable
            rows={rows(status.promotion_tests)}
            emptyTitle={String(promotion?.Promotion_Status ?? promotion?.promotion_status ?? "Research evidence pending")}
            emptyDetail="WRC, SPA, PBO, ICIR and downside gates are not fabricated when absent from the active artifact."
            maxRows={20}
          />
        </section>
        <section>
          <SectionHeading title="Data freshness" description="Every displayed result carries source recency and fallback status." />
          <DataTable
            rows={capabilityRows}
            columns={["Module", "Completeness", "Status", "Freshness_Requirement", "Missing_Evidence"]}
            emptyTitle="Capability matrix unavailable"
            emptyDetail="The dashboard can still render, but publication completeness cannot be audited."
            maxRows={20}
          />
          <DataTable
            rows={rows(status.data_freshness)}
            columns={["Namespace", "Status", "Age_Hours", "TTL_Hours", "Rows", "Fallback_Used"]}
            emptyTitle="Freshness report unavailable"
            emptyDetail="The prior active snapshot remains visible until a complete replacement passes publication checks."
            maxRows={20}
          />
        </section>
      </div>
    </div>
  );
}
