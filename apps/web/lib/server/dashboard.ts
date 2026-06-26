import "server-only";

import { promises as fs } from "node:fs";
import path from "node:path";

import type { DashboardArtifact, DashboardBundle, DashboardPayload, Row } from "@/lib/contracts";
import { rows, section } from "@/lib/contracts";
import { getServiceSupabase } from "@/lib/server/supabase";

const separator = "::";

function decodeArtifact(records: Row[], name: string): DashboardPayload | undefined {
  const payloads = new Map<string, unknown>();
  for (const row of records) {
    if (typeof row.artifact_name === "string") payloads.set(row.artifact_name, row.artifact_json);
  }
  function decode(key: string): unknown {
    const payload = payloads.get(key);
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
    const meta = payload as Record<string, unknown>;
    if (!meta._chunked) return payload;
    if (meta.schema === "dict_v1" || meta.schema === "top_level_dict_v1") {
      const keys = Array.isArray(meta.keys) ? meta.keys : Array.isArray(meta.sections) ? meta.sections : [];
      return Object.fromEntries(
        keys
          .map(String)
          .filter((child) => payloads.has(`${key}${separator}${child}`))
          .map((child) => [child, decode(`${key}${separator}${child}`)]),
      );
    }
    if (meta.schema === "list_v1") {
      const output: unknown[] = [];
      const parts = Number(meta.parts ?? 0);
      for (let index = 0; index < parts; index += 1) {
        const part = decode(`${key}${separator}part${String(index).padStart(4, "0")}`);
        if (Array.isArray(part)) output.push(...part);
      }
      return output;
    }
    return undefined;
  }
  const restored = decode(name);
  return restored && typeof restored === "object" && !Array.isArray(restored)
    ? (restored as DashboardPayload)
    : undefined;
}

function scopeOf(payload: DashboardPayload): DashboardArtifact["scope"] {
  const contract = payload.contract ?? {};
  if (contract.analytics_scope === "full_analysis") return "full_analysis";
  if (contract.analytics_scope === "market_snapshot" || contract.analytics_scope === "daily_snapshot") {
    return "daily_snapshot";
  }
  const snapshot = rows(section(payload, "status").snapshot_meta);
  if (String(snapshot[0]?.Snapshot_Mode ?? "").toLowerCase() === "daily_price_snapshot") return "daily_snapshot";
  const research = section(payload, "research");
  const allocationRows = rows(section(payload, "allocation").recommended_portfolio);
  const priceRows = rows(section(payload, "charts").price_paths);
  const riskRows = rows(section(payload, "tables").risk);
  const promotionRows = rows(section(payload, "status").promotion);
  const evidence =
    rows(section(payload, "tables").validation).length +
    rows(research.benchmark_governance).length +
    rows(research.variance_model_selection).length;
  const legacyFullEvidence =
    allocationRows.length >= 2 && priceRows.length >= 252 && riskRows.length >= 4 && promotionRows.length >= 1;
  return evidence > 0 || legacyFullEvidence ? "full_analysis" : "unknown";
}

function priceHistoryYears(payload: DashboardPayload | undefined): number | undefined {
  const priceRows = rows(section(payload, "charts").price_paths);
  const timestamps = priceRows
    .map((row) => Date.parse(String(row.Date ?? row.Observation_Date ?? row.date ?? "")))
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (timestamps.length < 2) return undefined;
  return (timestamps[timestamps.length - 1] - timestamps[0]) / (365.25 * 24 * 60 * 60 * 1000);
}

function evidenceNotices(full?: DashboardArtifact): string[] {
  if (!full) return ["The active full-analysis publication is unavailable; displaying the daily snapshot."];
  const years = priceHistoryYears(full.payload);
  if (years !== undefined && years < 3) {
    return [
      `Legacy full-analysis evidence spans ${years.toFixed(2)} years. The current publication contract requires at least 3.00 years; this artifact remains visible but is not promotion-eligible.`,
    ];
  }
  return [];
}

function meaningful(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value as Record<string, unknown>).length > 0;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
}

function nonEmptyEntries(sectionValue: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(sectionValue).filter(([, value]) => meaningful(value)));
}

function overlaySection(base: Record<string, unknown>, overlay: Record<string, unknown>): Record<string, unknown> {
  return {
    ...base,
    ...nonEmptyEntries(overlay),
  };
}

function mergeStatus(full: DashboardPayload, daily: DashboardPayload): NonNullable<DashboardPayload["status"]> {
  const fullStatus = section(full, "status");
  const dailyStatus = section(daily, "status");
  const output = { ...fullStatus };
  for (const key of ["snapshot_meta", "market_context", "data_freshness"]) {
    const value = dailyStatus[key];
    if (meaningful(value)) output[key] = value;
  }
  return output as NonNullable<DashboardPayload["status"]>;
}

function mergePayload(full?: DashboardPayload, daily?: DashboardPayload): DashboardPayload | undefined {
  if (!full) return daily;
  if (!daily) return full;
  const fullStrategy = section(full, "strategy_lab");
  return {
    ...full,
    contract: {
      ...section(full, "contract"),
      analytics_scope: "full_analysis_plus_daily_overlay",
      overlay_policy: "daily_market_sections_only",
    } as NonNullable<DashboardPayload["contract"]>,
    status: mergeStatus(full, daily),
    market_snapshot: overlaySection(
      section(full, "market_snapshot"),
      section(daily, "market_snapshot"),
    ) as NonNullable<DashboardPayload["market_snapshot"]>,
    market_intelligence: overlaySection(
      section(full, "market_intelligence"),
      section(daily, "market_intelligence"),
    ) as NonNullable<DashboardPayload["market_intelligence"]>,
    security_intelligence: overlaySection(
      section(full, "security_intelligence"),
      section(daily, "security_intelligence"),
    ) as NonNullable<DashboardPayload["security_intelligence"]>,
    fixed_income_intelligence: overlaySection(
      section(full, "fixed_income_intelligence"),
      section(daily, "fixed_income_intelligence"),
    ) as NonNullable<DashboardPayload["fixed_income_intelligence"]>,
    explanations: overlaySection(
      section(full, "explanations"),
      section(daily, "explanations"),
    ) as NonNullable<DashboardPayload["explanations"]>,
    strategy_lab: fullStrategy as NonNullable<DashboardPayload["strategy_lab"]>,
  };
}

async function loadFromSupabase(): Promise<DashboardBundle | null> {
  const client = getServiceSupabase();
  if (!client) return null;
  const pointerKeys = ["global:full_analysis", "global:daily_snapshot"];
  const { data: pointers, error: pointerError } = await client
    .from("publication_pointers")
    .select("pointer_key,publication_id,updated_at")
    .in("pointer_key", pointerKeys);
  if (pointerError || !pointers?.length) return null;
  const publicationIds = pointers.map((pointer) => pointer.publication_id);
  const { data: publications, error: publicationError } = await client
    .from("publication_manifests")
    .select("publication_id,run_id,publication_kind,activated_at")
    .in("publication_id", publicationIds);
  if (publicationError || !publications?.length) return null;

  const artifacts: DashboardArtifact[] = [];
  for (const publication of publications) {
    const { data: artifactRows, error } = await client
      .from("run_artifacts")
      .select("artifact_name,artifact_json,created_at")
      .eq("run_id", publication.run_id)
      .like("artifact_name", "dashboard_payload%");
    if (error || !artifactRows?.length) continue;
    const payload = decodeArtifact(artifactRows as Row[], "dashboard_payload");
    if (!payload) continue;
    artifacts.push({
      runId: String(publication.run_id),
      createdAt: String(publication.activated_at ?? artifactRows[0]?.created_at ?? ""),
      scope:
        publication.publication_kind === "full_analysis"
          ? "full_analysis"
          : publication.publication_kind === "daily_snapshot"
            ? "daily_snapshot"
            : scopeOf(payload),
      payload,
    });
  }
  const full = artifacts.find((artifact) => artifact.scope === "full_analysis");
  const daily = artifacts.find((artifact) => artifact.scope === "daily_snapshot");
  if (!full && !daily) return null;
  return {
    full,
    daily,
    merged: mergePayload(full?.payload, daily?.payload),
    source: "supabase",
    notices: evidenceNotices(full),
  };
}

async function loadLocalArtifacts(): Promise<DashboardBundle | null> {
  const localArtifactsAllowed =
    process.env.NODE_ENV !== "production" || process.env.QPK_ALLOW_LOCAL_ARTIFACTS === "1";
  if (!localArtifactsAllowed) return null;
  const rootCandidates = [
    process.cwd(),
    path.resolve(process.cwd(), "..", ".."),
  ];
  let root = rootCandidates[0];
  for (const candidate of rootCandidates) {
    try {
      await fs.access(path.join(candidate, ".quant_cache"));
      root = candidate;
      break;
    } catch {
      // Continue until a repository root containing the immutable artifact registry is found.
    }
  }
  const directory = path.join(root, ".quant_cache", "run_artifacts");
  let names: string[] = [];
  try {
    names = await fs.readdir(directory);
  } catch {
    // A current cloud snapshot can still be rendered when the historical registry is absent.
  }
  const candidates = await Promise.all(
    names
      .filter((name) => name.endsWith(".json") && !name.startsWith("atomic-run"))
      .map(async (name) => {
        const file = path.join(directory, name);
        return { file, stat: await fs.stat(file) };
      }),
  );
  candidates.sort((left, right) => right.stat.mtimeMs - left.stat.mtimeMs);
  let full: DashboardArtifact | undefined;
  let daily: DashboardArtifact | undefined;
  for (const candidate of candidates.slice(0, 25)) {
    const parsed = JSON.parse(await fs.readFile(candidate.file, "utf8")) as Record<string, unknown>;
    const payload = parsed.dashboard_payload as DashboardPayload | undefined;
    if (!payload) continue;
    const scope = scopeOf(payload);
    const artifact: DashboardArtifact = {
      runId: path.basename(candidate.file).split("_")[0],
      createdAt: candidate.stat.mtime.toISOString(),
      scope,
      payload,
    };
    if (scope === "full_analysis" && !full) full = artifact;
    if ((scope === "daily_snapshot" || scope === "unknown") && !daily) daily = artifact;
    if (full && daily) break;
  }
  const cloudDirectory = path.join(root, ".quant_cache", "cloud");
  async function readCloudArtifact(fileName: string): Promise<DashboardArtifact | undefined> {
    const file = path.join(cloudDirectory, fileName);
    try {
      const parsed = JSON.parse(await fs.readFile(file, "utf8")) as Record<string, unknown>;
      const payload = parsed.dashboard_payload as DashboardPayload | undefined;
      if (!payload) return undefined;
      const forcedScope = fileName.includes("full_analysis")
        ? "full_analysis"
        : fileName.includes("daily_snapshot")
          ? "daily_snapshot"
          : scopeOf(payload);
      return {
        runId: String(parsed.run_id ?? path.basename(fileName, ".json")),
        createdAt: String(parsed.created_at ?? ""),
        scope: forcedScope,
        payload,
      };
    } catch {
      return undefined;
    }
  }

  const cloudFull = await readCloudArtifact("latest_full_analysis_payload.json");
  if (cloudFull?.scope === "full_analysis" && !full) full = cloudFull;
  const cloudDaily = await readCloudArtifact("latest_daily_snapshot_payload.json");
  if (cloudDaily?.scope === "daily_snapshot") daily = cloudDaily;
  if (!daily) {
    const legacy = await readCloudArtifact("latest_dashboard_payload.json");
    if (legacy?.scope === "full_analysis" && !full) full = legacy;
    if (legacy && legacy.scope !== "full_analysis") daily = { ...legacy, scope: "daily_snapshot" };
  }
  if (!full && !daily) return null;
  return {
    full,
    daily,
    merged: mergePayload(full?.payload, daily?.payload),
    source: "local-artifact",
    notices: [
      "Development fallback: artifacts are loaded from the local immutable registry.",
      ...evidenceNotices(full),
    ],
  };
}

export async function loadDashboardBundle(): Promise<DashboardBundle> {
  try {
    const cloud = await loadFromSupabase();
    if (cloud) {
      if (process.env.NODE_ENV !== "production") {
        const local = await loadLocalArtifacts();
        const localStrategy = section(local?.daily?.payload, "strategy_lab");
        if (local?.daily && rows(localStrategy.summary).length) {
          return {
            full: cloud.full,
            daily: local.daily,
            merged: mergePayload(cloud.merged ?? cloud.full?.payload, local.daily.payload),
            source: "local-artifact",
            notices: [
              "Development overlay: the active cloud publication remains intact while the latest local daily artifact is rendered.",
              ...cloud.notices,
            ],
          };
        }
      }
      return cloud;
    }
  } catch (error) {
    console.error("Supabase dashboard read failed", error instanceof Error ? error.message : "unknown");
  }
  const local = await loadLocalArtifacts();
  if (local) return local;
  return {
    source: "unavailable",
    notices: [
      "No active publication is available. The prior snapshot was not replaced; verify the 07:00 CT publication job.",
    ],
  };
}
