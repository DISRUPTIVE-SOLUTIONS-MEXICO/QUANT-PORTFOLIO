"use client";

import { RotateCcw } from "lucide-react";
import { useEffect } from "react";

export default function ErrorPage({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <main className="error-page">
      <p className="eyebrow">Publication read error</p>
      <h1>The analytical snapshot could not be rendered.</h1>
      <p>The active database pointer was not modified. Retry the read or inspect Data Quality and publication logs.</p>
      <button type="button" onClick={reset}>
        <RotateCcw size={17} />
        Retry
      </button>
    </main>
  );
}
