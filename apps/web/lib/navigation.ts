import {
  Activity,
  BadgeDollarSign,
  BookOpenCheck,
  BrainCircuit,
  BriefcaseBusiness,
  ChartNoAxesCombined,
  CircleGauge,
  DatabaseZap,
  FlaskConical,
  Landmark,
  ScanSearch,
  ShieldCheck,
} from "lucide-react";

export const workspaces = [
  { slug: "", label: "Command Center", icon: CircleGauge, group: "Decision Center" },
  { slug: "market-intelligence", label: "Market Intelligence", icon: BrainCircuit, group: "Markets" },
  { slug: "rates-fixed-income", label: "Rates & Fixed Income", icon: Landmark, group: "Markets" },
  { slug: "equity-research", label: "Equity Research", icon: ScanSearch, group: "Markets" },
  { slug: "strategy-laboratory", label: "Strategy Laboratory", icon: FlaskConical, group: "Research" },
  { slug: "xcdr-research", label: "XCDR Research", icon: FlaskConical, group: "Research" },
  { slug: "validation-governance", label: "Validation & Governance", icon: ShieldCheck, group: "Research" },
  { slug: "portfolio-construction", label: "Portfolio Construction", icon: ChartNoAxesCombined, group: "Portfolio" },
  { slug: "risk-laboratory", label: "Risk Laboratory", icon: Activity, group: "Portfolio" },
  { slug: "my-portfolios", label: "My Portfolios", icon: BriefcaseBusiness, group: "Portfolio" },
  { slug: "paper-execution", label: "Paper Execution", icon: BadgeDollarSign, group: "Approval & Execution" },
  { slug: "data-quality", label: "Data Quality", icon: DatabaseZap, group: "Operations" },
  { slug: "administration", label: "Administration", icon: BookOpenCheck, group: "Operations" },
] as const;

export type WorkspaceSlug = (typeof workspaces)[number]["slug"];

export const workspaceGroups = [
  "Decision Center",
  "Markets",
  "Research",
  "Portfolio",
  "Approval & Execution",
  "Operations",
] as const;

export function workspaceBySlug(slug: string) {
  return workspaces.find((workspace) => workspace.slug === slug);
}
