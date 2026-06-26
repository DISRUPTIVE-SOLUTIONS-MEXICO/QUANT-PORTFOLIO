import { NextResponse } from "next/server";

import { loadDashboardBundle } from "@/lib/server/dashboard";

export const dynamic = "force-dynamic";

export async function GET() {
  const bundle = await loadDashboardBundle();
  return NextResponse.json(bundle, {
    headers: {
      "Cache-Control": "private, max-age=0, s-maxage=300, stale-while-revalidate=900",
    },
  });
}
