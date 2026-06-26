export type Scalar = string | number | boolean | null;
export type Row = Record<string, Scalar | Scalar[] | Record<string, Scalar>>;
export type Rows = Row[];

export interface DashboardPayload {
  contract?: Record<string, Scalar>;
  status?: Record<string, Rows | Row | Scalar>;
  allocation?: Record<string, Rows | Row | Scalar>;
  market_snapshot?: Record<string, Rows | Row | Scalar>;
  market_intelligence?: Record<string, Rows | Row | Scalar>;
  security_intelligence?: Record<string, Rows | Row | Scalar>;
  fixed_income_intelligence?: Record<string, Rows | Row | Scalar>;
  charts?: Record<string, Rows | Row | Scalar>;
  tables?: Record<string, Rows | Row | Scalar>;
  research?: Record<string, Rows | Row | Scalar>;
  strategy_lab?: Record<string, Rows | Row | Scalar>;
  diagnostics?: Record<string, Record<string, Rows>>;
  explanations?: Record<string, Scalar | Scalar[]>;
}

export interface DashboardArtifact {
  runId: string;
  createdAt?: string;
  scope: "daily_snapshot" | "full_analysis" | "unknown";
  payload: DashboardPayload;
}

export interface DashboardBundle {
  full?: DashboardArtifact;
  daily?: DashboardArtifact;
  merged?: DashboardPayload;
  source: "supabase" | "local-artifact" | "unavailable";
  notices: string[];
}

export interface PublicationRecord {
  publication_id: string;
  run_id: string;
  publication_kind: string;
  state: string;
  activated_at?: string;
}

export function rows(value: unknown): Rows {
  return Array.isArray(value) ? (value.filter((item) => item && typeof item === "object") as Rows) : [];
}

export function section(payload: DashboardPayload | undefined, key: keyof DashboardPayload): Record<string, unknown> {
  const value = payload?.[key];
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

export function finiteNumber(value: unknown): number {
  if (value === null || value === undefined || typeof value === "boolean") return Number.NaN;
  if (typeof value === "string" && value.trim() === "") return Number.NaN;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

export function metricMap(payload: DashboardPayload | undefined): Map<string, number> {
  const table = rows(section(payload, "tables").risk);
  const map = new Map<string, number>();
  for (const row of table) {
    const key = String(row.Metric ?? "");
    const value = finiteNumber(row.Value);
    if (key && Number.isFinite(value)) map.set(key, value);
  }
  return map;
}

export function normalizeSeriesLabel(label: string): string {
  const lower = label.toLowerCase();
  if (lower.includes("sortino") || lower.includes("daily causal allocation proxy")) {
    return "XCDR research portfolio price";
  }
  if (lower.includes("synthetic nav")) return "XCDR research portfolio price";
  return label;
}
