import { AppShell } from "@/components/app-shell";
import { CommandCenter } from "@/components/command-center";
import { loadDashboardBundle } from "@/lib/server/dashboard";
import { terminalContext } from "@/lib/terminal-context";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function HomePage() {
  const bundle = await loadDashboardBundle();
  return (
    <AppShell context={terminalContext(bundle)}>
      <CommandCenter bundle={bundle} />
    </AppShell>
  );
}
