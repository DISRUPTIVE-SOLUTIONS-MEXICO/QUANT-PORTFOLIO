"use client";

import { Play, RefreshCw } from "lucide-react";
import { FormEvent, useCallback, useEffect, useState } from "react";

import { getBrowserSupabase } from "@/lib/browser/supabase";
import type { Rows } from "@/lib/contracts";

import { DataTable } from "./data-table";

const DEFAULT_UNIVERSE = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, JPM, XOM, LLY, COST, LIN, NEE, DLR, SPY, QQQ";

async function accessToken(): Promise<string | null> {
  const client = getBrowserSupabase();
  if (!client) return null;
  const { data } = await client.auth.getSession();
  return data.session?.access_token ?? null;
}

export function OptimizationJobForm() {
  const [jobs, setJobs] = useState<Rows>([]);
  const [message, setMessage] = useState("Sign in to submit and monitor user-scoped optimization research.");
  const [submitting, setSubmitting] = useState(false);

  const loadJobs = useCallback(async () => {
    const token = await accessToken();
    if (!token) {
      setJobs([]);
      setMessage("Sign in to submit and monitor user-scoped optimization research.");
      return;
    }
    const response = await fetch("/api/jobs", { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) {
      setMessage("The job ledger could not be loaded. No research state was changed.");
      return;
    }
    const payload = (await response.json()) as { jobs?: Rows };
    setJobs(payload.jobs ?? []);
    setMessage(payload.jobs?.length ? "" : "No optimization jobs have been submitted.");
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadJobs(), 0);
    return () => window.clearTimeout(timer);
  }, [loadJobs]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    const form = new FormData(event.currentTarget);
    const token = await accessToken();
    if (!token) {
      setMessage("Sign in before submitting optimization research.");
      setSubmitting(false);
      return;
    }
    const tickers = String(form.get("tickers") ?? "")
      .split(/[\s,;]+/)
      .map((ticker) => ticker.trim())
      .filter(Boolean);
    const body = {
      portfolio_name: String(form.get("portfolio_name") ?? "XCDR Portfolio"),
      tickers,
      benchmark_ticker: String(form.get("benchmark_ticker") ?? "SPY"),
      filter_style: String(form.get("filter_style") ?? "factor"),
      objective: "xcdr_v3",
      base_period: String(form.get("base_period") ?? "3y"),
      initial_capital: Number(form.get("initial_capital") ?? 100000),
      monthly_contribution: Number(form.get("monthly_contribution") ?? 0),
      risk_aversion: Number(form.get("risk_aversion") ?? 5),
      max_drawdown: Number(form.get("max_drawdown") ?? 0.2),
      liquidity_need: String(form.get("liquidity_need") ?? "Medium"),
      base_currency: String(form.get("base_currency") ?? "USD"),
    };
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = (await response.json()) as { job_id?: string; error?: unknown };
      if (!response.ok) {
        setMessage(typeof payload.error === "string" ? payload.error : "The optimization job was rejected by the API contract.");
      } else {
        setMessage(`Optimization queued: ${payload.job_id ?? "job accepted"}. The current portfolio remains active until validation completes.`);
        await loadJobs();
      }
    } catch {
      setMessage("The optimization job could not be submitted. The current portfolio remains unchanged.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="optimization-console">
      <form onSubmit={submit}>
        <div className="optimization-grid">
          <label>
            Portfolio name
            <input name="portfolio_name" defaultValue="XCDR Portfolio" maxLength={64} required />
          </label>
          <label>
            Benchmark ξ
            <input name="benchmark_ticker" defaultValue="SPY" maxLength={16} required />
          </label>
          <label>
            Fundamental style
            <select name="filter_style" defaultValue="factor">
              <option value="factor">Multi-factor</option>
              <option value="growth">Growth</option>
              <option value="value">Value</option>
              <option value="quality">Quality</option>
              <option value="custom">Regime-adaptive</option>
            </select>
          </label>
          <label>
            Research history
            <select name="base_period" defaultValue="3y">
              <option value="3y">3 years</option>
              <option value="5y">5 years</option>
              <option value="10y">10 years</option>
            </select>
          </label>
          <label>
            Initial capital
            <input name="initial_capital" type="number" defaultValue={100000} min={1} max={1000000000} step={1000} required />
          </label>
          <label>
            Monthly contribution
            <input name="monthly_contribution" type="number" defaultValue={0} min={0} max={10000000} step={100} />
          </label>
          <label>
            Risk aversion · 0–10
            <input name="risk_aversion" type="number" defaultValue={5} min={0} max={10} step={0.5} required />
          </label>
          <label>
            Maximum drawdown budget
            <input name="max_drawdown" type="number" defaultValue={0.2} min={0.03} max={0.8} step={0.01} required />
          </label>
          <label>
            Liquidity need
            <select name="liquidity_need" defaultValue="Medium">
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
          </label>
          <label>
            Base currency
            <select name="base_currency" defaultValue="USD">
              <option value="USD">USD</option>
              <option value="MXN">MXN</option>
              <option value="EUR">EUR</option>
              <option value="GBP">GBP</option>
            </select>
          </label>
        </div>
        <label className="universe-field">
          Research universe
          <textarea name="tickers" defaultValue={DEFAULT_UNIVERSE} rows={4} required />
          <span>Comma or space separated. The backend normalizes, deduplicates and applies liquidity and data-coverage gates.</span>
        </label>
        <div className="optimization-actions">
          <button type="submit" className="primary-action" disabled={submitting}>
            <Play size={16} aria-hidden="true" />
            {submitting ? "Queueing research…" : "Run XCDR research"}
          </button>
          <button type="button" onClick={() => void loadJobs()}>
            <RefreshCw size={16} aria-hidden="true" />
            Refresh job ledger
          </button>
          <span role="status">{message}</span>
        </div>
      </form>
      <DataTable
        rows={jobs}
        columns={["job_id", "job_type", "status", "result_run_id", "created_at", "started_at", "finished_at", "error"]}
        emptyTitle="Optimization job ledger"
        emptyDetail={message}
        maxRows={50}
        exportName="optimization-jobs"
      />
    </div>
  );
}
