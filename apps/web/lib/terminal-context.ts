import type { DashboardBundle } from "@/lib/contracts";
import { rows, section } from "@/lib/contracts";

export interface TerminalContext {
  benchmark: string;
  evidence: string;
  baseCurrency: string;
  asOf?: string;
  source: string;
  publication: string;
}

export function terminalContext(bundle: DashboardBundle): TerminalContext {
  const payload = bundle.merged;
  const strategy = section(payload, "strategy_lab");
  const research = section(payload, "research");
  const status = section(payload, "status");
  const registry = rows(research.model_registry);
  const market = rows(status.market_context);
  const benchmark = String(
    strategy.benchmark_xi ??
      registry[0]?.benchmark_ticker ??
      registry[0]?.Benchmark ??
      market[0]?.Benchmark_Ticker ??
      market[0]?.Benchmark ??
      "ξ",
  );
  const asOf = bundle.daily?.createdAt ?? bundle.full?.createdAt;
  return {
    benchmark,
    evidence: bundle.full ? "OOS + holdout" : "Daily market snapshot",
    baseCurrency: String(market[0]?.Base_Currency ?? market[0]?.Currency ?? "USD"),
    asOf,
    source:
      bundle.source === "supabase"
        ? "Supabase registry"
        : bundle.source === "local-artifact"
          ? "Local registry"
          : "Unavailable",
    publication:
      bundle.full && bundle.daily
        ? "Full research + daily overlay"
        : bundle.full
          ? "Full research"
          : bundle.daily
            ? "Daily overlay only"
            : "No active publication",
  };
}
