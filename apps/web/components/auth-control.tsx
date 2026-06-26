"use client";

import { LogIn, LogOut } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { getBrowserSupabase } from "@/lib/browser/supabase";

export function AuthControl() {
  const client = getBrowserSupabase();
  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    if (!client) return;
    void client.auth.getSession().then(({ data }) => setEmail(data.session?.user.email ?? null));
    const { data } = client.auth.onAuthStateChange((_event, session) => setEmail(session?.user.email ?? null));
    return () => data.subscription.unsubscribe();
  }, [client]);

  if (!client) return <span className="auth-state muted">Auth not configured</span>;
  if (!email) {
    return (
      <Link className="auth-action" href="/sign-in">
        <LogIn size={16} aria-hidden="true" />
        Sign in
      </Link>
    );
  }
  return (
    <button
      className="auth-action"
      type="button"
      onClick={() => {
        void client.auth.signOut();
      }}
      title={email}
    >
      <LogOut size={16} aria-hidden="true" />
      Sign out
    </button>
  );
}
