import { NextResponse } from "next/server";

import manifest from "../../../../../FEATURE_PRESERVATION_MANIFEST.json";

export const revalidate = 3600;

export async function GET() {
  return NextResponse.json(manifest, {
    headers: {
      "Cache-Control": "public, max-age=300, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
