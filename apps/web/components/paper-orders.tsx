"use client";

import { useEffect, useState } from "react";

import { getBrowserSupabase } from "@/lib/browser/supabase";
import type { Rows } from "@/lib/contracts";

import { DataTable } from "./data-table";

export function PaperOrders() {
  const [orders, setOrders] = useState<Rows>([]);
  const [state, setState] = useState("Loading paper order blotter...");

  useEffect(() => {
    void (async () => {
      const client = getBrowserSupabase();
      const { data } = client ? await client.auth.getSession() : { data: { session: null } };
      const token = data.session?.access_token;
      if (!token) {
        setState("Sign in to view paper order intents and pre-trade decisions.");
        return;
      }
      const response = await fetch("/api/orders", { headers: { Authorization: `Bearer ${token}` } });
      if (!response.ok) {
        setState("The paper blotter could not be loaded. No order was submitted.");
        return;
      }
      const payload = (await response.json()) as { orders?: Rows };
      setOrders(payload.orders ?? []);
      setState(payload.orders?.length ? "" : "No paper order intents exist.");
    })();
  }, []);

  return (
    <DataTable
      rows={orders}
      columns={["order_intent_id", "status", "run_id", "portfolio_version_id", "created_at", "approved_at"]}
      emptyTitle="Paper execution blotter"
      emptyDetail={state}
    />
  );
}
