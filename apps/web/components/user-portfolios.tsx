"use client";

import { useEffect, useState } from "react";

import { getBrowserSupabase } from "@/lib/browser/supabase";
import type { Rows } from "@/lib/contracts";

import { DataTable } from "./data-table";

async function bearerToken() {
  const client = getBrowserSupabase();
  if (!client) return null;
  const { data } = await client.auth.getSession();
  return data.session?.access_token ?? null;
}

export function UserPortfolios() {
  const [portfolios, setPortfolios] = useState<Rows>([]);
  const [state, setState] = useState("Loading authenticated portfolio catalog...");

  useEffect(() => {
    void (async () => {
      const token = await bearerToken();
      if (!token) {
        setState("Sign in to view user-scoped portfolio versions.");
        return;
      }
      const response = await fetch("/api/portfolios", { headers: { Authorization: `Bearer ${token}` } });
      if (!response.ok) {
        setState("Portfolio catalog could not be loaded. Verify the Supabase migration and RLS policies.");
        return;
      }
      const payload = (await response.json()) as { portfolios?: Rows };
      setPortfolios(payload.portfolios ?? []);
      setState(payload.portfolios?.length ? "" : "No saved portfolio versions yet.");
    })();
  }, []);

  return (
    <DataTable
      rows={portfolios}
      columns={["name", "base_currency", "active_version_id", "updated_at"]}
      emptyTitle="User portfolio catalog"
      emptyDetail={state}
    />
  );
}
