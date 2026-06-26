import Link from "next/link";

import { SignInForm } from "@/components/sign-in-form";

export const metadata = { title: "Sign in" };

export default function SignInPage() {
  return (
    <main className="auth-page">
      <section className="auth-panel">
        <span className="eyebrow">Quant Portfolio-Kaizen</span>
        <h1>Institutional workspace access</h1>
        <p>
          Supabase Auth protects user portfolios, jobs and paper-order evidence. Shared market publications remain
          read-only.
        </p>
        <SignInForm />
        <Link href="/">Return to Command Center</Link>
      </section>
    </main>
  );
}
