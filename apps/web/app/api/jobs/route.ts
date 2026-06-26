import { NextResponse } from "next/server";
import { z } from "zod";

import { requireUser } from "@/lib/server/auth";
import { getUserSupabase } from "@/lib/server/supabase";

const jobSchema = z.object({
  tickers: z.array(z.string().trim().min(1).max(16)).min(2).max(150),
  benchmark_ticker: z.string().trim().min(1).max(16),
  filter_style: z.enum(["growth", "value", "quality", "factor", "custom"]),
  objective: z.literal("xcdr_v3"),
  base_period: z.enum(["3y", "5y", "10y"]),
  portfolio_name: z.string().trim().min(1).max(64).optional(),
  initial_capital: z.number().positive().max(1_000_000_000).optional(),
  monthly_contribution: z.number().min(0).max(10_000_000).optional(),
  risk_aversion: z.number().min(0).max(10).optional(),
  max_drawdown: z.number().min(0.03).max(0.80).optional(),
  liquidity_need: z.enum(["Low", "Medium", "High"]).optional(),
  base_currency: z.string().trim().length(3).optional(),
});

export async function GET(request: Request) {
  try {
    const { userId, accessToken } = await requireUser(request);
    const client = getUserSupabase(accessToken);
    if (!client) return NextResponse.json({ error: "Authentication is not configured." }, { status: 503 });
    const url = new URL(request.url);
    const jobId = z.string().uuid().safeParse(url.searchParams.get("job_id"));
    let query = client
      .from("jobs")
      .select("job_id,job_type,status,result_run_id,error,created_at,started_at,finished_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(50);
    if (jobId.success) query = query.eq("job_id", jobId.data);
    const { data, error } = await query;
    if (error) return NextResponse.json({ error: "Job query failed." }, { status: 500 });
    return NextResponse.json({ jobs: data ?? [] });
  } catch (error) {
    const code = error instanceof Error ? error.message : "UNAUTHORIZED";
    return NextResponse.json({ error: code }, { status: code === "UNAUTHORIZED" ? 401 : 503 });
  }
}

export async function POST(request: Request) {
  try {
    const { userId, accessToken } = await requireUser(request);
    const parsed = jobSchema.safeParse(await request.json());
    if (!parsed.success) return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    const client = getUserSupabase(accessToken);
    if (!client) return NextResponse.json({ error: "Authentication is not configured." }, { status: 503 });
    const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const { data: recentJobs, error: quotaError } = await client
      .from("jobs")
      .select("job_id,status,created_at")
      .eq("user_id", userId)
      .eq("job_type", "optimization")
      .gte("created_at", since)
      .order("created_at", { ascending: false })
      .limit(10);
    if (quotaError) return NextResponse.json({ error: "Job quota could not be verified." }, { status: 500 });
    if ((recentJobs ?? []).some((job) => ["queued", "running"].includes(String(job.status)))) {
      return NextResponse.json({ error: "An optimization job is already queued or running." }, { status: 409 });
    }
    if ((recentJobs ?? []).length >= 3) {
      return NextResponse.json({ error: "Daily optimization research limit reached." }, { status: 429 });
    }
    const normalizedTickers = [...new Set(parsed.data.tickers.map((ticker) => ticker.toUpperCase().replaceAll(".", "-")))];
    if (normalizedTickers.length < 2) {
      return NextResponse.json({ error: "At least two distinct tickers are required." }, { status: 400 });
    }
    const normalizedConfig = {
      ...parsed.data,
      tickers: normalizedTickers,
      benchmark_ticker: parsed.data.benchmark_ticker.toUpperCase().replaceAll(".", "-"),
      base_currency: parsed.data.base_currency?.toUpperCase(),
    };
    const { data, error } = await client
      .from("jobs")
      .insert({
        user_id: userId,
        job_type: "optimization",
        status: "queued",
        config: normalizedConfig,
      })
      .select("job_id,status")
      .single();
    if (error) return NextResponse.json({ error: "Optimization job could not be queued." }, { status: 500 });
    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    const code = error instanceof Error ? error.message : "UNAUTHORIZED";
    return NextResponse.json({ error: code }, { status: code === "UNAUTHORIZED" ? 401 : 503 });
  }
}
