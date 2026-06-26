export default function Loading() {
  return (
    <main className="loading-page" aria-live="polite">
      <div className="loading-rule" />
      <strong>Resolving the active immutable publication</strong>
      <span>The prior validated snapshot remains authoritative during refresh.</span>
    </main>
  );
}
