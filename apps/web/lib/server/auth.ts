import "server-only";

import { getUserSupabase } from "@/lib/server/supabase";

export async function requireUser(request: Request): Promise<{ userId: string; accessToken: string }> {
  const header = request.headers.get("authorization") ?? "";
  const accessToken = header.startsWith("Bearer ") ? header.slice(7).trim() : "";
  if (!accessToken) throw new Error("UNAUTHORIZED");
  const client = getUserSupabase(accessToken);
  if (!client) throw new Error("AUTH_NOT_CONFIGURED");
  const { data, error } = await client.auth.getUser(accessToken);
  if (error || !data.user) throw new Error("UNAUTHORIZED");
  return { userId: data.user.id, accessToken };
}
