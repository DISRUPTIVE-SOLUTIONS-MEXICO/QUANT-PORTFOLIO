import { NextResponse } from "next/server";

import { requireUser } from "@/lib/server/auth";
import { getUserSupabase } from "@/lib/server/supabase";

export async function GET(request: Request) {
  try {
    const { userId, accessToken } = await requireUser(request);
    const client = getUserSupabase(accessToken);
    if (!client) return NextResponse.json({ error: "Authentication is not configured." }, { status: 503 });
    const { data, error } = await client
      .from("user_portfolios")
      .select("portfolio_id,name,base_currency,active_version_id,created_at,updated_at")
      .eq("user_id", userId)
      .order("updated_at", { ascending: false });
    if (error) return NextResponse.json({ error: "Portfolio query failed." }, { status: 500 });
    return NextResponse.json({ portfolios: data ?? [] });
  } catch (error) {
    const code = error instanceof Error ? error.message : "UNAUTHORIZED";
    return NextResponse.json({ error: code }, { status: code === "UNAUTHORIZED" ? 401 : 503 });
  }
}
