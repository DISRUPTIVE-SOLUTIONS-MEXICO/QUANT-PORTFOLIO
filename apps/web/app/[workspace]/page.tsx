import { notFound } from "next/navigation";

import { AppShell } from "@/components/app-shell";
import { WorkspaceView } from "@/components/workspace-view";
import { workspaceBySlug } from "@/lib/navigation";
import { loadDashboardBundle } from "@/lib/server/dashboard";
import { terminalContext } from "@/lib/terminal-context";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function WorkspacePage({ params }: { params: Promise<{ workspace: string }> }) {
  const { workspace } = await params;
  if (!workspaceBySlug(workspace)) notFound();
  const bundle = await loadDashboardBundle();
  return (
    <AppShell context={terminalContext(bundle)}>
      <WorkspaceView slug={workspace} bundle={bundle} />
    </AppShell>
  );
}
