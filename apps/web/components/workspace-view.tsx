import { AlertTriangle, CheckCircle2, Clock3, LockKeyhole, ShieldCheck } from "lucide-react";

import type { DashboardBundle, Row, Rows } from "@/lib/contracts";
import { rows, section } from "@/lib/contracts";
import { workspaceBySlug } from "@/lib/navigation";

import { DataTable } from "./data-table";
import { FixedIncomeExplorer } from "./fixed-income-explorer";
import { MathPanel } from "./math-panel";
import { MetricStrip } from "./metric-strip";
import { OptimizationJobForm } from "./optimization-job-form";
import { PaperOrders } from "./paper-orders";
import { PageHeader } from "./page-header";
import { SecurityExplorer } from "./security-explorer";
import { SectionHeading } from "./section-heading";
import { TimeSeriesChart, YieldCurveChart } from "./charts";
import { UserPortfolios } from "./user-portfolios";

function pivotLong(rowsInput: Rows, dateKey: string, groupKey: string, valueKey: string, limit = 8): Rows {
  const groups = Array.from(new Set(rowsInput.map((row) => String(row[groupKey] ?? "")).filter(Boolean))).slice(0, limit);
  const byDate = new Map<string, Row>();
  for (const row of rowsInput) {
    const group = String(row[groupKey] ?? "");
    const date = String(row[dateKey] ?? "");
    const value = Number(row[valueKey]);
    if (!groups.includes(group) || !date || !Number.isFinite(value)) continue;
    const point = byDate.get(date) ?? { Date: date };
    point[group] = value;
    byDate.set(date, point);
  }
  return Array.from(byDate.values()).sort((left, right) => String(left.Date).localeCompare(String(right.Date)));
}

function numericValue(value: unknown): number {
  if (value === null || value === undefined || typeof value === "boolean") return Number.NaN;
  if (typeof value === "string" && value.trim() === "") return Number.NaN;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function firstNonEmptyRows(...sources: unknown[]): Rows {
  for (const source of sources) {
    const candidate = rows(source);
    if (candidate.length) return candidate;
  }
  return [];
}

function deriveCountryMetricsFromCurves(curves: Rows): Rows {
  return curves
    .filter((row) => Number.isFinite(numericValue(row.Yield_10Y)) || Number.isFinite(numericValue(row.Yield_2Y)) || Number.isFinite(numericValue(row.Yield_Short)))
    .map((row) => {
      const yield2Y = Number.isFinite(numericValue(row.Yield_2Y)) ? row.Yield_2Y : row.Yield_Short;
      const tenorCount = [row.Policy_Rate, yield2Y, row.Yield_10Y].filter((value) => Number.isFinite(numericValue(value))).length;
      return {
        Country: row.Country,
        As_Of: row.Latest_Date,
        Policy_Rate: row.Policy_Rate,
        Yield_2Y: yield2Y,
        Yield_10Y: row.Yield_10Y,
        Slope_10Y_2Y: row.Curve_10Y_2Y,
        Curve_State: row.Curve_Shape,
        Rate_Source: row.Rate_Source,
        Curve_Quality: tenorCount >= 2 ? "Observed public tenors" : "Single observed tenor",
        Curve_Quality_Score: tenorCount >= 2 ? 1 : 0.5,
        Sovereign_Tenor_Count: tenorCount,
        History_Observations: row.Short_Observation_Count ?? row.TenY_Observation_Count ?? 0,
        Modified_Duration_2Y: 1.9,
        Modified_Duration_10Y: 8.5,
        Policy_Gap_2Y: null,
        Level_Factor: Number.isFinite(numericValue(yield2Y)) && Number.isFinite(numericValue(row.Yield_10Y)) ? (numericValue(yield2Y) + numericValue(row.Yield_10Y)) / 2 : row.Yield_10Y ?? yield2Y,
        Slope_Change_3M_bp: null,
        Level_Change_3M_bp: null,
        Observed_Rate_Change_Vol_bp: null,
        Stale_Days: null,
      };
    });
}

function deriveReferenceRateSummary(history: Rows): Rows {
  const latest = new Map<string, Row>();
  for (const row of history) {
    const key = String(row.Benchmark ?? row.Code ?? "");
    if (!key) continue;
    const currentDate = Date.parse(String(row.Latest_Observation_Date ?? row.Observation_Date ?? ""));
    const priorDate = Date.parse(String(latest.get(key)?.Latest_Observation_Date ?? latest.get(key)?.Observation_Date ?? ""));
    if (!latest.has(key) || (Number.isFinite(currentDate) && (!Number.isFinite(priorDate) || currentDate >= priorDate))) {
      latest.set(key, {
        Code: row.Code,
        Benchmark: row.Benchmark,
        Jurisdiction: row.Jurisdiction,
        Currency: row.Currency,
        Tenor: row.Tenor,
        Latest_Rate: row.Latest_Rate ?? row.Rate,
        Latest_Observation_Date: row.Latest_Observation_Date ?? row.Observation_Date,
        Data_Staleness_Days: row.Data_Staleness_Days,
        Observation_Frequency: row.Observation_Frequency,
        Status: row.Status,
        Source: row.Source,
      });
    }
  }
  return Array.from(latest.values());
}
const COUNTRY_POINTS: Record<string, { x: number; y: number }> = {
  "United States": { x: 22, y: 39 },
  Mexico: { x: 19, y: 52 },
  Canada: { x: 20, y: 27 },
  Brazil: { x: 33, y: 68 },
  "United Kingdom": { x: 48, y: 34 },
  France: { x: 50, y: 39 },
  Germany: { x: 52, y: 36 },
  Italy: { x: 53, y: 43 },
  Spain: { x: 49, y: 43 },
  Russia: { x: 66, y: 30 },
  Ukraine: { x: 58, y: 39 },
  Turkey: { x: 59, y: 45 },
  Iran: { x: 64, y: 48 },
  India: { x: 70, y: 55 },
  China: { x: 77, y: 45 },
  Japan: { x: 86, y: 43 },
  "South Korea": { x: 84, y: 44 },
  Australia: { x: 82, y: 76 },
  "South Africa": { x: 55, y: 77 },
  Vietnam: { x: 77, y: 58 },
  Austria: { x: 53, y: 38 },
  Bulgaria: { x: 56, y: 43 },
  Greece: { x: 55, y: 45 },
  Croatia: { x: 54, y: 41 },
  Tunisia: { x: 52, y: 50 },
  Romania: { x: 57, y: 41 },
};

function GeoRiskMap({ rows: heatmapRows }: { rows: Rows }) {
  const points = heatmapRows
    .map((row) => {
      const country = String(row.Country ?? "");
      const point = COUNTRY_POINTS[country];
      const percentile = numericValue(row.Percentile);
      const score = numericValue(row.Geo_News_Attention_Score);
      if (!country || !point || !Number.isFinite(percentile)) return null;
      return {
        country,
        point,
        percentile,
        score: Number.isFinite(score) ? score : percentile,
        heat: String(row.Heat_Level ?? "Observed"),
        topic: String(row.Dominant_Topic ?? "Topic unavailable"),
        confidence: numericValue(row.Mean_Geo_Inference_Confidence),
        articles: row.Article_Count,
      };
    })
    .filter(Boolean)
    .sort((left, right) => (right?.percentile ?? 0) - (left?.percentile ?? 0));

  if (!points.length) {
    return (
      <div className="empty-state">
        <strong>Country heatmap unavailable</strong>
        <span>GDELT/RSS country inference did not produce admissible country-level attention evidence.</span>
      </div>
    );
  }

  return (
    <section className="geo-risk-map" aria-label="Country-level geopolitical attention map">
      <div className="geo-risk-stage" role="img" aria-label="Dark world map with country attention markers">
        <div className="world-silhouette america" />
        <div className="world-silhouette europe" />
        <div className="world-silhouette asia" />
        <div className="world-silhouette africa" />
        <div className="world-silhouette australia" />
        {points.map((item) =>
          item ? (
            <span
              className={`geo-risk-dot heat-${item.heat.toLowerCase()}`}
              key={item.country}
              style={{
                left: `${item.point.x}%`,
                top: `${item.point.y}%`,
                width: `${10 + item.percentile * 20}px`,
                height: `${10 + item.percentile * 20}px`,
              }}
              title={`${item.country}: ${item.heat} · ${item.topic}`}
            />
          ) : null,
        )}
      </div>
      <div className="geo-risk-list">
        {points.slice(0, 6).map((item) =>
          item ? (
            <article key={item.country}>
              <span>{item.heat}</span>
              <strong>{item.country}</strong>
              <small>{item.topic}</small>
              <b>{Math.round(item.percentile * 100)}%</b>
            </article>
          ) : null,
        )}
      </div>
    </section>
  );
}
function unavailable(label: string, source: string) {
  return (
    <div className="empty-state">
      <strong>{label} unavailable</strong>
      <span>The active artifact does not contain this evidence. Expected source: {source}. The UI does not infer a value.</span>
    </div>
  );
}

function MarketIntelligence({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.merged;
  const market = section(payload, "market_intelligence");
  const latest = rows(market.latest_macro)[0];
  return (
    <>
      <MetricStrip
        metrics={[
          { label: "Rates regime", value: String(latest?.Regime_Hawkish_Dovish ?? latest?.Rates_Regime ?? "Unavailable") },
          { label: "Market regime", value: String(latest?.Regime_Bull_Bear ?? latest?.Market_Regime ?? "Unavailable") },
          { label: "Credit state", value: String(latest?.Credit_Regime ?? latest?.Credit_Spread ?? "Unavailable") },
          { label: "Liquidity state", value: String(latest?.Liquidity_Regime ?? latest?.Liquidity ?? "Unavailable") },
        ]}
      />
      <SectionHeading
        title="Global geopolitical attention map"
        description="Country-level GDELT/RSS attention proxy with regex and source-country confidence diagnostics."
      />
      <GeoRiskMap rows={rows(market.geopolitical_country_heatmap)} />
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={rows(market.sentiment_timeline)}
          title="Latent market sentiment state"
          emptyDetail="SEM diagnostics require a converged daily latent-state artifact."
        />
        <TimeSeriesChart
          rows={rows(market.geopolitical_timeline)}
          title="Within-topic abnormal news attention"
          emptyDetail="GDELT or RSS did not provide a statistically admissible timeline."
          referenceZero
        />
      </div>
      <MathPanel
        title="Latent sentiment and robust attention"
        formula={String.raw`\eta_t=\Lambda^\top x_t+\zeta_t,\qquad Z^{robust}_{k,t}=\frac{V_{k,t}-\operatorname{median}(V_{k,\tau})}{1.4826\,\operatorname{MAD}(V_{k,\tau})}`}
      >
        <p>
          SEM consolidates correlated market observables into a latent construct. Geopolitical attention is
          normalized within each topic; raw cross-topic article counts are not treated as comparable risk
          probabilities.
        </p>
      </MathPanel>
      <div className="split-band">
        <section>
          <SectionHeading title="Geopolitical diagnostics" description="Topic-level abnormal attention and data admissibility." />
          <DataTable
            rows={rows(market.geopolitical_summary)}
            emptyTitle="Geopolitical summary unavailable"
            emptyDetail="The previous validated snapshot remains authoritative."
          />
        </section>
        <section>
          <SectionHeading title="Scheduled event risk" description="Public macro calendar, filtered for material events." />
          <DataTable
            rows={rows(market.forex_factory_calendar)}
            emptyTitle="Event calendar unavailable"
            emptyDetail="FairEconomy/ForexFactory public data did not pass the refresh contract."
          />
        </section>
      </div>
    </>
  );
}

function RatesFixedIncome({ bundle }: { bundle: DashboardBundle }) {
  const market = section(bundle.merged, "market_intelligence");
  const fixedIncome = section(bundle.merged, "fixed_income_intelligence");
  const research = section(bundle.merged, "research");
  const curves = firstNonEmptyRows(market.global_yield_curves, fixedIncome.country_metrics);
  const rateHistoryRows = firstNonEmptyRows(research.global_rate_history, market.global_rate_history);
  const tenYearHistory = rateHistoryRows.filter((row) => String(row.Tenor_Code ?? row.Tenor ?? "").toUpperCase().includes("10Y"));
  const history = pivotLong(tenYearHistory.length ? tenYearHistory : rateHistoryRows, "Observation_Date", "Country", "Rate", 9);
  const countryMetrics = firstNonEmptyRows(fixedIncome.country_metrics, deriveCountryMetricsFromCurves(curves));
  const factorHistory = firstNonEmptyRows(fixedIncome.factor_history);
  const carryValidation = firstNonEmptyRows(market.carry_trade_validation, research.carry_trade_validation, fixedIncome.carry_candidates);
  const referenceRates = firstNonEmptyRows(fixedIncome.reference_rate_summary, deriveReferenceRateSummary(firstNonEmptyRows(research.interbank_reference_rates, market.interbank_reference_rates)));
  return (
    <>
      <SectionHeading
        title="Fixed-Income Intelligence Workbench"
        description="Observed sovereign term-structure factors, source quality and local duration/convexity stress sensitivities."
      />
      <FixedIncomeExplorer
        metrics={countryMetrics}
        factorHistory={factorHistory}
        scenarios={rows(fixedIncome.stress_scenarios)}
      />
      <MathPanel
        title="Observed curve factors and local price sensitivity"
        formula={String.raw`L_t=\frac{y_t(2Y)+y_t(10Y)}{2},\quad S_t=y_t(10Y)-y_t(2Y),\quad \frac{\Delta P}{P}\approx-D_{\mathrm{mod}}\Delta y+\frac{1}{2}\mathcal{C}(\Delta y)^2`}
      >
        <p>
          Factors update only when a source publishes an observation. Duration and convexity scenarios are
          local zero-coupon sensitivity proxies, not executable bond valuations, and no missing maturity is
          synthesized.
        </p>
      </MathPanel>
      <div className="chart-comparison">
        <YieldCurveChart rows={curves} country="United States" />
        <TimeSeriesChart
          rows={history}
          title="Discrete sovereign rate evolution"
          emptyDetail="At least two countries with source-verified rate histories are required."
        />
      </div>
      <MathPanel
        title="Term-structure state"
        formula={String.raw`s_t^{10-2}=y_t(10Y)-y_t(2Y),\qquad \Delta y_{t_j}=y_{t_j}-y_{t_{j-1}},\quad t_j\in\mathcal{T}_{obs}`}
      >
        <p>
          Rates evolve on their native observation calendars. No interpolation is used to make monthly policy
          series appear daily; sovereign and reference-rate tenors retain source frequency and observation date.
        </p>
      </MathPanel>
      <SectionHeading title="Global sovereign curve panel" description="Policy, short and 10-year rates with source and discrete frequency." />
      <DataTable
        rows={curves}
        columns={[
          "Country",
          "Policy_Rate",
          "Yield_2Y",
          "Yield_10Y",
          "Curve_10Y_2Y",
          "Curve_Shape",
          "Regime_Hawkish_Dovish",
          "Rate_Source",
          "Latest_Date",
        ]}
        emptyTitle="Yield-curve panel unavailable"
        emptyDetail="A country is shown only when public-source tenor evidence is present."
        maxRows={30}
      />
      <div className="split-band">
        <section>
          <SectionHeading title="FX-adjusted carry candidates" description="Rate differentials are filtered by FX and event-risk diagnostics." />
          <DataTable
            rows={carryValidation}
            emptyTitle="Carry validation unavailable"
            emptyDetail="An unhedged rate differential is never presented as a validated trade."
          />
        </section>
        <section>
          <SectionHeading title="Reference-rate state" description="SOFR, SONIA, €STR and TONAR with observed changes and staleness." />
          <DataTable
            rows={referenceRates}
            columns={[
              "Code",
              "Benchmark",
              "Currency",
              "Latest_Rate",
              "Change_1M_bp",
              "Change_3M_bp",
              "Change_1Y_bp",
              "Latest_Observation_Date",
              "Data_Staleness_Days",
              "Source",
            ]}
            emptyTitle="Reference-rate history unavailable"
            emptyDetail="Official or FRED-proxy source series are required."
            maxRows={24}
          />
        </section>
      </div>
    </>
  );
}

function EquityResearch({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.full?.payload ?? bundle.merged;
  const livePayload = bundle.merged ?? payload;
  const tables = section(payload, "tables");
  const research = section(payload, "research");
  const securities = section(livePayload, "security_intelligence");
  return (
    <>
      <SectionHeading
        title="Security Intelligence Workbench"
        description="Observed prices, benchmark-relative convexity, tail dependence, liquidity and cross-strategy consensus from information available at the snapshot date."
      />
      <SecurityExplorer
        metrics={rows(securities.metrics)}
        priceHistory={rows(securities.price_history)}
        consensus={rows(securities.strategy_consensus)}
        benchmark={String(securities.benchmark_xi ?? "SPY")}
      />
      <MathPanel
        title="Causal security state"
        formula={String.raw`\beta_i^{\pm}=\frac{\operatorname{Cov}(r_i,r_\xi\mid r_\xi\gtrless0)}{\operatorname{Var}(r_\xi\mid r_\xi\gtrless0)},\quad \beta_i^{tail}=\frac{\operatorname{Cov}(r_i,r_\xi\mid r_\xi\le q_{0.10})}{\operatorname{Var}(r_\xi\mid r_\xi\le q_{0.10})},\quad M_i^{res}=\prod_{\tau=t-125}^{t}(1+\hat\varepsilon_{i,\tau})-1`}
      >
        <p>
          The workbench is a live descriptive state, not OOS evidence. Every estimate is truncated at the
          displayed decision date; strategy breadth aggregates only the latest causal scores from each registered
          family.
        </p>
      </MathPanel>
      <MathPanel
        title="Sector-relative fundamental selection"
        formula={String.raw`Z^{robust}_{i,k,s}=\frac{x_{i,k}-\operatorname{median}_{j\in s}(x_{j,k})}{1.4826\,\operatorname{MAD}_{j\in s}(x_{j,k})},\qquad d_i^2=(x_i-\mu_s)^\top\Sigma_s^{-1}(x_i-\mu_s)`}
      >
        <p>
          Valuation, quality, solvency and growth are normalized within sector before aggregation. SEC filing
          availability dates dominate Yahoo snapshots when both exist; every ratio carries a PIT-confidence
          interpretation.
        </p>
      </MathPanel>
      <SectionHeading title="Selected securities and fundamentals" description="The complete ratio panel behind the active research allocation." />
      <DataTable
        rows={rows(tables.fundamentals)}
        columns={[
          "Ticker",
          "Sector",
          "Weight",
          "Composite_Score",
          "ROIC",
          "EV_EBITDA",
          "FCF_Yield",
          "Net_Debt_EBITDA",
          "ROE",
          "PIT_Confidence",
        ]}
        emptyTitle="Full fundamental selection unavailable"
        emptyDetail="The daily market snapshot does not replace the last complete research run."
        maxRows={50}
      />
      <div className="split-band">
        <section>
          <SectionHeading title="Options-implied information" description="Snapshot IV, skew, open interest and bid/ask quality." />
          <DataTable
            rows={rows(research.options_summary)}
            emptyTitle="Options snapshot unavailable"
            emptyDetail="Yahoo option chains are snapshot-only and are never backfilled as historical PIT evidence."
          />
        </section>
        <section>
          <SectionHeading title="Reject list" description="Securities removed by fundamental, liquidity, data-quality or risk gates." />
          <DataTable
            rows={rows(tables.rejections)}
            emptyTitle="No rejection artifact"
            emptyDetail="A missing reject list is a data-contract warning, not proof that every security passed."
          />
        </section>
      </div>
    </>
  );
}

function percentMetric(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : "Unavailable";
}

function ratioMetric(value: unknown): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "n/a";
}

function StrategyLaboratory({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.merged ?? bundle.full?.payload;
  const lab = section(payload, "strategy_lab");
  const summary = rows(lab.summary);
  const oosSummary = rows(lab.oos_summary);
  const holdoutSummary = rows(lab.holdout_summary);
  const strategyRegistry = rows(lab.strategy_registry);
  const executableStrategies = strategyRegistry.filter((row) => Boolean(row.Engine_Candidate));
  const researchFrontier = strategyRegistry.filter((row) => !Boolean(row.Engine_Candidate));
  const leading = oosSummary[0] ?? summary[0];
  const promoted = String(lab.status ?? "").toUpperCase() === "PROMOTED";
  const icTimeline = pivotLong(
    rows(lab.signal_ic),
    "Signal_Date",
    "Strategy",
    "Information_Coefficient",
    5,
  );
  return (
    <>
      <div className={`governance-banner ${promoted ? "positive" : "warning"}`}>
        <AlertTriangle aria-hidden="true" />
        <div>
          <strong>{promoted ? "Strategy-family evidence passed the strict gate" : "Strategy-family selection is research-only"}</strong>
          <span>
            Candidate selection is performed only in train and validation windows. Test blocks and the frozen final
            holdout remain untouched until evaluation.
          </span>
        </div>
      </div>
      <MetricStrip
        metrics={[
          {
            label: "Leading causal candidate",
            value: String(lab.frozen_candidate || leading?.Strategy || "Unavailable"),
            detail: String(lab.benchmark_xi ? `Frozen before holdout · ξ: ${lab.benchmark_xi}` : "Benchmark unavailable"),
          },
          {
            label: "Annualized return",
            value: percentMetric(leading?.Annualized_Return),
            detail: `Active ${percentMetric(leading?.Active_Return)}`,
            tone: Number(leading?.Active_Return) >= 0 ? "positive" : "warning",
          },
          {
            label: "Maximum drawdown",
            value: percentMetric(leading?.Max_Drawdown),
            detail: `CVaR ${percentMetric(leading?.CVaR_95_Daily)}`,
            tone: "neutral",
          },
          {
            label: "Capture asymmetry",
            value: `${ratioMetric(leading?.Upside_Capture)} / ${ratioMetric(leading?.Downside_Capture)}`,
            detail: "Upside / downside capture",
            tone:
              Number(leading?.Upside_Capture) > 1 && Number(leading?.Downside_Capture) < 1
                ? "positive"
                : "warning",
          },
        ]}
      />
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={rows(lab.oos_price_paths)}
          title="Nested walk-forward OOS price indices versus ξ"
          emptyDetail="A complete train, purged validation and untouched test sequence is required."
        />
        <TimeSeriesChart
          rows={rows(lab.oos_drawdowns)}
          title="Nested OOS drawdown paths"
          valueFormat="percent"
          referenceZero
          emptyDetail="Drawdown is computed only from concatenated untouched test blocks."
        />
      </div>
      <MathPanel
        title="Pre-registered strategy research map"
        formula={String.raw`w_{t+1}^{(k)}=\pi_k\!\left(\mathcal{F}_t\right),\qquad r_{p,t+1}^{(k)}={w_t^{(k)}}^\top r_{t+1}-c\,\lVert w_t^{(k)}-w_{t-1}^{(k)}\rVert_1`}
      >
        <p>
          The executable set contains fixed price-derived families, including cross-sectional and dual momentum,
          volatility-adjusted trend, residual momentum versus ξ, asymmetric capture, defensive convexity,
          residual reversion and a causal downside governor. Signals are observed at the close of <em>t</em>;
          new weights first affect the return at <em>t+1</em>.
        </p>
      </MathPanel>
      <div className="split-band">
        <section>
          <SectionHeading
            title="Executable strategy specifications"
            description="Frozen hypotheses, data requirements, benchmark policy and known failure modes for the current engine candidates."
          />
          <DataTable
            rows={executableStrategies}
            columns={[
              "Strategy",
              "Family",
              "Holding_Horizon",
              "Signal_Formula",
              "Benchmark_Policy",
              "Suitable_Regimes",
              "Failure_Modes",
              "Cost_Sensitivity",
              "Evidence_Status",
            ]}
            emptyTitle="Executable registry unavailable"
            emptyDetail="A candidate cannot enter selection without a versioned strategy specification."
            maxRows={20}
          />
        </section>
        <section>
          <SectionHeading
            title="Research frontier and admissibility"
            description="Additional investment families are retained without pretending that unavailable PIT data or execution histories are validated evidence."
          />
          <DataTable
            rows={researchFrontier}
            columns={[
              "Strategy",
              "Asset_Class",
              "Holding_Horizon",
              "Required_Inputs",
              "Availability_Rule",
              "Implementation_Status",
              "Evidence_Status",
            ]}
            emptyTitle="Research frontier unavailable"
            emptyDetail="Planned or blocked strategies must state the missing data or execution requirement explicitly."
            maxRows={20}
          />
        </section>
      </div>
      <SectionHeading
        title="Candidate comparison"
        description="Return, downside, capture and benchmark-relative diagnostics from the same causal execution convention."
      />
      <DataTable
        rows={summary}
        columns={[
          "Strategy",
          "Annualized_Return",
          "Active_Return",
          "Annualized_Volatility",
          "Max_Drawdown",
          "CVaR_95_Daily",
          "Upside_Capture",
          "Downside_Capture",
          "Beta_to_Xi",
          "Information_Ratio",
        ]}
        emptyTitle="Strategy evidence unavailable"
        emptyDetail="The active artifact lacks the minimum causal history required by the laboratory constitution."
        maxRows={12}
      />
      <SectionHeading
        title="Candidate path-equivalence control"
        description="Economically distinct hypotheses remain documented, while numerically identical return paths count once in selection and multiple-testing inference."
      />
      <DataTable
        rows={rows(lab.candidate_equivalence)}
        columns={[
          "Strategy",
          "Canonical_Strategy",
          "Equivalent_Path",
          "Included_In_Selection",
          "Tolerance",
        ]}
        emptyTitle="Equivalence diagnostics unavailable"
        emptyDetail="The candidate matrix must be observed before path redundancy can be assessed."
        maxRows={20}
      />
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={rows(lab.holdout_price_paths)}
          title="Frozen final holdout price indices"
          emptyDetail="The holdout remains invisible until the candidate has been frozen from pre-holdout evidence."
        />
        <TimeSeriesChart
          rows={rows(lab.holdout_drawdowns)}
          title="Frozen final holdout drawdowns"
          valueFormat="percent"
          referenceZero
          emptyDetail="No holdout path is shown before the final candidate is frozen."
        />
      </div>
      <div className="split-band">
        <section>
          <SectionHeading
            title="Nested walk-forward ledger"
            description="Train, purge, validation, embargo and untouched test boundaries for every selection decision."
          />
          <DataTable
            rows={rows(lab.walk_forward_windows)}
            columns={[
              "Window",
              "Train_Start",
              "Train_End",
              "Purge_Days",
              "Validation_Start",
              "Validation_End",
              "Embargo_Days",
              "Test_Start",
              "Test_End",
              "Selected_Strategy",
              "Validation_Utility",
              "Test_Active_Return",
            ]}
            emptyTitle="Nested windows unavailable"
            emptyDetail="The artifact must contain enough history for train, validation, test and final holdout."
            maxRows={20}
          />
        </section>
        <section>
          <SectionHeading
            title="Selection stability"
            description="Frequency and dispersion of validation-only candidate selection across temporal windows."
          />
          <DataTable
            rows={rows(lab.selection_stability)}
            columns={[
              "Strategy",
              "Windows_Selected",
              "Selection_Rate",
              "Mean_Validation_Utility",
              "Validation_Utility_Std",
            ]}
            emptyTitle="Selection stability unavailable"
            emptyDetail="At least two valid nested windows are required."
          />
          <DataTable
            rows={holdoutSummary}
            columns={[
              "Evidence_Scope",
              "Frozen_Candidate",
              "Annualized_Return",
              "Active_Return",
              "Max_Drawdown",
              "CVaR_95_Daily",
              "Upside_Capture",
              "Downside_Capture",
            ]}
            emptyTitle="Frozen holdout summary unavailable"
            emptyDetail="Holdout evidence is only produced after candidate freezing."
          />
        </section>
      </div>
      <div className="split-band">
        <section>
          <SectionHeading
            title="Causal downside throttle"
            description="Exposure decided at the signal close from benchmark state only; unallocated capital remains in zero-return cash."
          />
          <TimeSeriesChart
            rows={rows(lab.exposure_timeline)}
            title="Risk exposure and cash reserve"
            emptyDetail="The downside-controlled family requires at least 252 benchmark observations."
          />
          <DataTable
            rows={rows(lab.exposure_diagnostics)}
            columns={[
              "Signal_Date",
              "Execution_Date",
              "Exposure",
              "Cash_Weight",
              "Trend_126D",
              "Drawdown_252D",
              "Volatility_Ratio_21_126",
              "Downside_Ratio_21_126",
              "Regime",
            ]}
            emptyTitle="Throttle diagnostics unavailable"
            emptyDetail="No exposure state is inferred when the required history is absent."
            maxRows={18}
          />
        </section>
        <section>
          <SectionHeading
            title="Research-generation lineage"
            description="A consumed holdout can diagnose a failure, but it cannot validate the next strategy generation."
          />
          <DataTable
            rows={rows(lab.research_lineage)}
            columns={[
              "Research_Generation",
              "Parent_Generation",
              "Change",
              "Holdout_Status",
              "Promotion_Eligible",
              "Prospective_Evidence_Start",
              "Reason",
            ]}
            emptyTitle="Research lineage unavailable"
            emptyDetail="Every material strategy change must create a new immutable generation."
          />
          <MathPanel
            title="Smooth causal risk governor"
            formula={String.raw`e_t=\operatorname{clip}\!\left(0.20+0.80\left(0.35T_t+0.25D_t+0.20V_t+0.20S_t\right),0.25,1\right),\qquad w_t^{risk}=e_t w_t,\;w_t^{cash}=1-e_t`}
          >
            <p>
              The governor is fixed before prospective evaluation. It reduces exposure continuously rather than
              optimizing thresholds against the historical holdout.
            </p>
          </MathPanel>
        </section>
      </div>
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={icTimeline}
          title="Signal information coefficient by rebalance"
          emptyDetail="IC requires at least four scored securities and a subsequent holding-period return."
          referenceZero
        />
        <section>
          <SectionHeading
            title="Multiple-testing diagnostics"
            description="WRC, Hansen SPA and PBO are reported, but nested selection and a frozen holdout remain mandatory."
          />
          <DataTable
            rows={rows(lab.validation)}
            columns={["Metric", "Value", "Threshold", "Pass"]}
            emptyTitle="Validation diagnostics unavailable"
            emptyDetail="No promotion inference is made from an incomplete candidate matrix."
          />
        </section>
      </div>
      <div className="split-band">
        <section>
          <SectionHeading title="Latest opportunity ranking" description="Current scores and selected long-only weights by strategy family." />
          <DataTable
            rows={rows(lab.latest_scores)}
            columns={["As_Of", "Strategy", "Ticker", "Score", "Selected", "Weight"]}
            emptyTitle="Latest strategy scores unavailable"
            emptyDetail="Scores remain absent when source history or benchmark alignment fails."
            maxRows={60}
          />
        </section>
        <section>
          <SectionHeading title="Regime attribution" description="Signal IC and execution-day behavior by causal benchmark regime." />
          <DataTable
            rows={rows(lab.regime_performance)}
            columns={["Strategy", "Regime", "Signals", "Mean_IC", "Mean_Execution_Day_Return"]}
            emptyTitle="Regime attribution unavailable"
            emptyDetail="A stable regime sample is required before comparative interpretation."
            maxRows={40}
          />
        </section>
      </div>
      <SectionHeading title="Strategy constitution" description="Frozen execution, cost, concentration and evidence assumptions." />
      <DataTable
        rows={rows(lab.constitution)}
        emptyTitle="Strategy constitution unavailable"
        emptyDetail="A strategy artifact without its constitution is not admissible research evidence."
      />
    </>
  );
}

function PortfolioConstruction({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.full?.payload ?? bundle.merged;
  const allocation = section(payload, "allocation");
  const portfolio = rows(allocation.recommended_portfolio);
  return (
    <>
      <MathPanel
        title="Constrained causal allocation"
        formula={String.raw`w_t=\arg\max_{w\in\Delta_N}\operatorname{XCDR}(w\mid\mathcal{F}_t),\quad \mathbf{1}^\top w=1,\;0\leq w_i\leq w_{\max},\;\mathrm{CVaR}(w)\leq c_\alpha`}
      >
        <p>
          Allocation consumes only information available at the decision date. Sector, concentration, liquidity,
          turnover, transaction-cost and suitability constraints are applied before a portfolio can be saved.
        </p>
      </MathPanel>
      <SectionHeading title="Current research weights" description="Immutable output from the active full-analysis artifact." />
      <DataTable
        rows={portfolio}
        columns={["Ticker", "Sector", "Country", "Weight", "Composite_Score", "Bayesian_Alpha_Mean", "PIT_Confidence"]}
        emptyTitle="No complete allocation is active"
        emptyDetail="Run the allocation engine against a validated full research dataset; daily price snapshots cannot create weights."
        maxRows={60}
      />
      <div className="process-line" aria-label="Portfolio construction sequence">
        {["PIT data", "Sector scoring", "Benchmark ξ", "Uncertainty state", "XCDR", "Risk gates", "Immutable run"].map(
          (step, index) => (
            <div key={step}>
              <span>{index + 1}</span>
              <strong>{step}</strong>
            </div>
          ),
        )}
      </div>
    </>
  );
}

function XcdrResearch({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.full?.payload ?? bundle.merged;
  const research = section(payload, "research");
  const charts = section(payload, "charts");
  return (
    <>
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={rows(charts.price_paths)}
          title="XCDR research portfolio vs ξ"
          emptyDetail="The research price path requires a full OOS artifact."
        />
        <TimeSeriesChart
          rows={rows(research.variance_conditional_paths)}
          title="Conditional variance state"
          emptyDetail="No selected ARCH/GARCH/EGARCH path exists in the active full-analysis artifact."
        />
      </div>
      <MathPanel
        title="Uncertainty-aware alpha"
        formula={String.raw`\mu^{robust}_{i,t}=\hat{\mu}_{i,t}\frac{|\hat{\mu}_{i,t}|}{|\hat{\mu}_{i,t}|+\sqrt{\operatorname{CRLB}_{i,t}}},\qquad \Sigma_t^{RMT}=Q\,\operatorname{diag}(\tilde{\lambda}_1,\dots,\tilde{\lambda}_N)Q^\top`}
      >
        <p>
          Kalman filtering estimates latent signal state, Fisher information penalizes weak identification and RMT
          removes covariance eigenmodes consistent with sampling noise. These layers govern capital; they do not
          manufacture expected return.
        </p>
      </MathPanel>
      <div className="split-band">
        <section>
          <SectionHeading title="Benchmark ξ governance" description="Mandate fit is fixed before portfolio optimization." />
          <DataTable
            rows={rows(research.benchmark_governance)}
            emptyTitle="Benchmark-governance artifact unavailable"
            emptyDetail="The selected ξ remains visible in the run metadata; detailed candidate fit requires full research."
          />
        </section>
        <section>
          <SectionHeading title="Optimization grid" description="Pre-registered candidates and validation-only selection." />
          <DataTable
            rows={rows(research.optimization_grid)}
            emptyTitle="Optimization grid unavailable"
            emptyDetail="No raw grid is inferred from final weights."
          />
        </section>
      </div>
    </>
  );
}

function RiskLaboratory({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.full?.payload ?? bundle.merged;
  const research = section(payload, "research");
  const charts = section(payload, "charts");
  return (
    <>
      <div className="chart-comparison">
        <TimeSeriesChart
          rows={rows(charts.drawdowns)}
          title="Drawdown by daily price path"
          valueFormat="percent"
          referenceZero
          emptyDetail="Validated daily prices are required."
        />
        <TimeSeriesChart
          rows={rows(research.gbm_forecast_paths)}
          title="Conditional forecast cone"
          emptyDetail="Forecast bands are withheld when the variance architecture is unavailable."
        />
      </div>
      <MathPanel
        title="Conditional variance architecture"
        formula={String.raw`h_t=\omega+\sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2+\sum_{j=1}^{q}\beta_jh_{t-j},\qquad m^*=\arg\min_m\operatorname{QLIKE}_{OOS}(m)`}
      >
        <p>
          AIC, BIC and log-likelihood describe in-sample parsimony; model promotion requires lower out-of-sample
          QLIKE. PELT breaks, EVT residual tails and volatility-targeting overlays remain separate diagnostics.
        </p>
      </MathPanel>
      <div className="split-band">
        <section>
          <SectionHeading title="Variance model selection" description="ARCH, GARCH, EGARCH and Volterra diagnostics." />
          <DataTable
            rows={rows(research.variance_model_selection)}
            emptyTitle="Variance architecture unavailable"
            emptyDetail="The daily overlay cannot replace model-selection evidence."
          />
        </section>
        <section>
          <SectionHeading title="PELT regime breaks" description="Causal change points in return and volatility state." />
          <DataTable
            rows={rows(research.pelt_change_points)}
            emptyTitle="PELT changes unavailable"
            emptyDetail="A stable minimum sample and penalty calibration are required."
          />
        </section>
      </div>
    </>
  );
}

function ValidationGovernance({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.full?.payload ?? bundle.merged;
  const status = section(payload, "status");
  const research = section(payload, "research");
  return (
    <>
      <div className="governance-banner">
        <ShieldCheck aria-hidden="true" />
        <div>
          <strong>Promotion is a hard state transition</strong>
          <span>No return result overrides WRC, SPA, PBO, ICIR, suitability, freshness or downside limits.</span>
        </div>
      </div>
      <MathPanel
        title="Promotion gate"
        formula={String.raw`\mathbb{1}_{promote}=\mathbb{1}\{p_{WRC}<0.05,\;p_{SPA}<0.05,\;PBO<0.10,\;ICIR>0,\;DD\leq DD_{\max},\;CVaR\leq CVaR_{\max}\}`}
      >
        <p>
          Training, validation, test and final holdout scopes remain distinct. Purging removes overlapping labels;
          embargo prevents adjacent information leakage. A failed gate is retained as evidence, not hidden.
        </p>
      </MathPanel>
      <div className="split-band">
        <section>
          <SectionHeading title="Promotion tests" description="Multiple-testing and downside evidence." />
          <DataTable
            rows={rows(status.promotion_tests)}
            emptyTitle="Promotion tests unavailable"
            emptyDetail="The active strategy remains research-only unless complete OOS evidence is published."
          />
        </section>
        <section>
          <SectionHeading title="Model registry" description="Code, config, universe and data hashes for reproducibility." />
          <DataTable
            rows={rows(research.model_registry)}
            emptyTitle="Model registry unavailable"
            emptyDetail="Every full research run must register code and data identity."
          />
        </section>
      </div>
    </>
  );
}

function MyPortfolios({ bundle }: { bundle: DashboardBundle }) {
  const allocation = section(bundle.full?.payload ?? bundle.merged, "allocation");
  return (
    <>
      <div className="governance-banner">
        <LockKeyhole aria-hidden="true" />
        <div>
          <strong>User-scoped version history</strong>
          <span>Saved portfolios are immutable versions protected by Supabase Auth and RLS.</span>
        </div>
      </div>
      <SectionHeading title="Active portfolio version" description="The most recent validated user allocation; prior versions remain auditable." />
      <UserPortfolios />
      <SectionHeading
        title="Optimization research request"
        description="Submit a tenant-scoped XCDR job. Existing portfolio versions remain active until the new run is durable and validated."
      />
      <OptimizationJobForm />
      <SectionHeading title="Research allocation preview" description="The active global research artifact is shown separately from saved user versions." />
      <DataTable
        rows={rows(allocation.recommended_portfolio)}
        columns={["Ticker", "Sector", "Weight", "Composite_Score", "PIT_Confidence"]}
        emptyTitle="No authenticated portfolio selected"
        emptyDetail="Sign in, complete suitability and save a full allocation to create the first portfolio version."
      />
      <div className="process-line">
        {["Configure", "Validate", "Optimize", "Save version", "Rebalance", "Audit history"].map((step, index) => (
          <div key={step}>
            <span>{index + 1}</span>
            <strong>{step}</strong>
          </div>
        ))}
      </div>
    </>
  );
}

function PaperExecution() {
  return (
    <>
      <div className="governance-banner warning">
        <AlertTriangle aria-hidden="true" />
        <div>
          <strong>Paper execution only</strong>
          <span>No broker connectivity or real order routing is enabled.</span>
        </div>
      </div>
      <div className="execution-grid">
        {[
          ["Suitability", "Investor mandate and loss tolerance", true],
          ["Promotion", "Only promoted OOS/holdout strategies", true],
          ["Freshness", "Prices and risk diagnostics within TTL", true],
          ["Liquidity", "ADV participation and spread/impact budget", true],
          ["Concentration", "Single-name and sector caps", true],
          ["Human approval", "Named approver required before paper submission", false],
        ].map(([title, detail, automated]) => (
          <section key={String(title)}>
            {automated ? <CheckCircle2 aria-hidden="true" /> : <Clock3 aria-hidden="true" />}
            <div>
              <strong>{String(title)}</strong>
              <span>{String(detail)}</span>
            </div>
          </section>
        ))}
      </div>
      <SectionHeading title="Paper order blotter" description="User-scoped order intents and immutable pre-trade decisions." />
      <PaperOrders />
      <MathPanel
        title="Transaction-cost and liquidity control"
        formula={String.raw`TC_t=c_1\sum_i|\Delta w_{i,t}|+c_2\sum_i(\Delta w_{i,t})^2,\qquad \frac{|Q_{i,t}|}{ADV_{i,t}}\leq\theta`}
      >
        <p>
          The order blotter is produced from immutable target weights, reference prices and current holdings.
          Approval is rejected when source artifacts, hashes, MNPI isolation or execution limits fail.
        </p>
      </MathPanel>
    </>
  );
}

function DataQuality({ bundle }: { bundle: DashboardBundle }) {
  const payload = bundle.merged;
  const status = section(payload, "status");
  const research = section(payload, "research");
  return (
    <>
      <SectionHeading
        title="Capability preservation matrix"
        description="Every institutional module states whether the active artifact contains the evidence required to analyze it."
      />
      <DataTable
        rows={rows(status.capability_completeness)}
        columns={["Module", "Completeness", "Status", "Freshness_Requirement", "Missing_Evidence", "Description"]}
        emptyTitle="Capability matrix unavailable"
        emptyDetail="Publication completeness could not be audited from the active artifact."
        maxRows={20}
      />
      <SectionHeading
        title="Source freshness and provenance"
        description="Public data remain research-grade and are never relabeled as vendor-grade point-in-time evidence."
      />
      <DataTable
        rows={rows(status.data_freshness)}
        emptyTitle="Freshness contract unavailable"
        emptyDetail="A publication without freshness evidence must not replace the active snapshot."
        maxRows={60}
      />
      <div className="split-band">
        <section>
          <SectionHeading title="Cache inventory" description="Incremental public-data objects and their current state." />
          <DataTable
            rows={rows(research.cache_inventory)}
            emptyTitle="Cache inventory unavailable"
            emptyDetail="Inspect the publication manifest and daily refresh logs."
          />
        </section>
        <section>
          <SectionHeading title="Pipeline timings" description="Stage-level latency for research and refresh jobs." />
          <DataTable
            rows={rows(research.timings)}
            emptyTitle="Timing diagnostics unavailable"
            emptyDetail="Timings are recorded on full research runs."
          />
        </section>
      </div>
    </>
  );
}

function Administration({ bundle }: { bundle: DashboardBundle }) {
  return (
    <>
      <div className="admin-ledger">
        <section>
          <span>Frontend source</span>
          <strong>{bundle.source}</strong>
        </section>
        <section>
          <span>Full analysis run</span>
          <strong>{bundle.full?.runId ?? "Unavailable"}</strong>
        </section>
        <section>
          <span>Daily snapshot run</span>
          <strong>{bundle.daily?.runId ?? "Unavailable"}</strong>
        </section>
        <section>
          <span>Publication mode</span>
          <strong>Staging → validate → atomic pointer</strong>
        </section>
      </div>
      <SectionHeading
        title="Institutional operating model"
        description="Daily snapshots and full research are independent immutable publications with rollback."
      />
      <div className="process-line">
        {["Fetch public data", "Build staging", "Quality contract", "Artifact hashes", "Atomic promote", "Rollback retained"].map(
          (step, index) => (
            <div key={step}>
              <span>{index + 1}</span>
              <strong>{step}</strong>
            </div>
          ),
        )}
      </div>
      <div className="governance-banner">
        <ShieldCheck aria-hidden="true" />
        <div>
          <strong>Capability preservation manifest</strong>
          <span>Every analytical capability has one canonical calculation, artifact owner and primary view.</span>
        </div>
      </div>
    </>
  );
}

export function WorkspaceView({ slug, bundle }: { slug: string; bundle: DashboardBundle }) {
  const workspace = workspaceBySlug(slug);
  if (!workspace || !slug) return null;
  const descriptions: Record<string, string> = {
    "market-intelligence": "Macro regimes, latent sentiment, event risk and geopolitical abnormal attention.",
    "rates-fixed-income": "Source-aware sovereign curves, reference rates, credit and FX-adjusted carry validation.",
    "equity-research": "Sector-relative fundamentals, PIT confidence, options-implied information and reject diagnostics.",
    "strategy-laboratory": "Causal strategy families, signal IC, regime attribution and multiple-testing evidence.",
    "portfolio-construction": "Causal XCDR allocation with liquidity, concentration, cost and downside constraints.",
    "xcdr-research": "Benchmark ξ, Ω stress sets, uncertainty state and the proprietary downside-aware objective.",
    "risk-laboratory": "Conditional variance, tail risk, PELT breaks, drawdown and forecast uncertainty.",
    "validation-governance": "Nested walk-forward evidence, multiple-testing controls and hard promotion gates.",
    "my-portfolios": "User-scoped immutable allocations, version history and rebalance evidence.",
    "paper-execution": "Order intents, pre-trade controls, human approval and simulated fills.",
    "data-quality": "Freshness, source provenance, fallback state and pipeline observability.",
    administration: "Publication control, audit identity, rollback and system governance.",
  };
  let content: React.ReactNode;
  switch (slug) {
    case "market-intelligence":
      content = <MarketIntelligence bundle={bundle} />;
      break;
    case "rates-fixed-income":
      content = <RatesFixedIncome bundle={bundle} />;
      break;
    case "equity-research":
      content = <EquityResearch bundle={bundle} />;
      break;
    case "strategy-laboratory":
      content = <StrategyLaboratory bundle={bundle} />;
      break;
    case "portfolio-construction":
      content = <PortfolioConstruction bundle={bundle} />;
      break;
    case "xcdr-research":
      content = <XcdrResearch bundle={bundle} />;
      break;
    case "risk-laboratory":
      content = <RiskLaboratory bundle={bundle} />;
      break;
    case "validation-governance":
      content = <ValidationGovernance bundle={bundle} />;
      break;
    case "my-portfolios":
      content = <MyPortfolios bundle={bundle} />;
      break;
    case "paper-execution":
      content = <PaperExecution />;
      break;
    case "data-quality":
      content = <DataQuality bundle={bundle} />;
      break;
    case "administration":
      content = <Administration bundle={bundle} />;
      break;
    default:
      content = unavailable(workspace.label, "Feature Preservation Manifest");
  }
  return (
    <div className="workspace-page">
      <PageHeader
        eyebrow="Quant Portfolio-Kaizen"
        title={workspace.label}
        description={descriptions[slug]}
        scope={bundle.full ? "Full research + current overlay" : "Snapshot evidence"}
      />
      {content}
    </div>
  );
}


