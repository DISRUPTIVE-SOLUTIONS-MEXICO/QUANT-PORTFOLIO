import { NextResponse } from "next/server";
import { z } from "zod";

import { requireUser } from "@/lib/server/auth";
import { getUserSupabase } from "@/lib/server/supabase";

const pretradeJobSchema = z.object({
  portfolio_version_id: z.string().uuid(),
  portfolio_value: z.number().positive().max(1_000_000_000),
  current_weights: z.record(z.string().trim().min(1).max(16), z.number().min(0).max(1)).default({}),
  acknowledgement: z.literal("paper_execution_only"),
});

export async function GET(request: Request) {
  try {
    const { userId, accessToken } = await requireUser(request);
    const client = getUserSupabase(accessToken);
    if (!client) return NextResponse.json({ error: "Authentication is not configured." }, { status: 503 });
    const { data, error } = await client
      .from("order_intents")
      .select(
        "order_intent_id,run_id,portfolio_version_id,status,contract_json,created_at,approved_by,approved_at,pretrade_decisions(decision_id,approved,contract_json,evaluated_at)",
      )
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(50);
    if (error) return NextResponse.json({ error: "Order-intent query failed." }, { status: 500 });
    return NextResponse.json({ orders: data ?? [] });
  } catch (error) {
    const code = error instanceof Error ? error.message : "UNAUTHORIZED";
    return NextResponse.json({ error: code }, { status: code === "UNAUTHORIZED" ? 401 : 503 });
  }
}

export async function POST(request: Request) {
  try {
    const { userId, accessToken } = await requireUser(request);
    const parsed = pretradeJobSchema.safeParse(await request.json());
    if (!parsed.success) return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    const client = getUserSupabase(accessToken);
    if (!client) return NextResponse.json({ error: "Authentication is not configured." }, { status: 503 });

    const { data: version, error: versionError } = await client
      .from("portfolio_versions")
      .select("version_id")
      .eq("version_id", parsed.data.portfolio_version_id)
      .eq("user_id", userId)
      .maybeSingle();
    if (versionError || !version) {
      return NextResponse.json({ error: "Portfolio version is unavailable to this user." }, { status: 404 });
    }

    const { data, error } = await client
      .from("jobs")
      .insert({
        user_id: userId,
        job_type: "paper_pretrade",
        status: "queued",
        config: parsed.data,
      })
      .select("job_id,status")
      .single();
    if (error) return NextResponse.json({ error: "Pre-trade evaluation could not be queued." }, { status: 500 });
    return NextResponse.json(data, { status: 202 });
  } catch (error) {
    const code = error instanceof Error ? error.message : "UNAUTHORIZED";
    return NextResponse.json({ error: code }, { status: code === "UNAUTHORIZED" ? 401 : 503 });
  }
}
