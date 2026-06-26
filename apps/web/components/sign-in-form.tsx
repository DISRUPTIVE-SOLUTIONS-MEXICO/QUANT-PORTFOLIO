"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import { getBrowserSupabase } from "@/lib/browser/supabase";

export function SignInForm() {
  const router = useRouter();
  const client = getBrowserSupabase();
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    if (!client) {
      setError("Supabase authentication is not configured.");
      return;
    }
    const form = new FormData(event.currentTarget);
    setPending(true);
    const { error: authError } = await client.auth.signInWithPassword({
      email: String(form.get("email") ?? ""),
      password: String(form.get("password") ?? ""),
    });
    setPending(false);
    if (authError) {
      setError("Sign-in failed. Verify the email, password and Supabase user status.");
      return;
    }
    router.replace("/my-portfolios");
  }

  return (
    <form className="sign-in-form" onSubmit={submit}>
      <label>
        Email
        <input name="email" type="email" autoComplete="email" required />
      </label>
      <label>
        Password
        <input name="password" type="password" autoComplete="current-password" required />
      </label>
      {error ? <div className="form-error">{error}</div> : null}
      <button className="primary-action" type="submit" disabled={pending}>
        {pending ? "Authenticating..." : "Sign in"}
      </button>
    </form>
  );
}
