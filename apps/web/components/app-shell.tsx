"use client";

import { ChevronLeft, Menu, RadioTower, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

import type { TerminalContext } from "@/lib/terminal-context";
import { workspaceGroups, workspaces } from "@/lib/navigation";

import { AuthControl } from "./auth-control";

interface AppShellProps {
  children: React.ReactNode;
  context: TerminalContext;
}

export function AppShell({ children, context }: AppShellProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  return (
    <div className={`app-frame ${collapsed ? "sidebar-collapsed" : ""}`}>
      <a className="skip-link" href="#main-content">
        Skip to analysis
      </a>
      <header className="topbar">
        <button
          className="icon-button mobile-menu"
          type="button"
          aria-label="Open navigation"
          onClick={() => setMobileOpen(true)}
        >
          <Menu size={20} />
        </button>
        <div className="brand-lockup">
          <span className="brand-mark">QK</span>
          <div>
            <strong>Quant Portfolio-Kaizen</strong>
            <span>Institutional research and pre-trade control</span>
          </div>
        </div>
        <div className="publication-state">
          <RadioTower size={16} aria-hidden="true" />
          <span>{context.publication}</span>
          <time>
            {context.asOf
              ? new Date(context.asOf).toLocaleString("en-US", { timeZone: "America/Mexico_City" })
              : "n/a"}
          </time>
        </div>
        <AuthControl />
      </header>

      <section className="contextbar" aria-label="Active analytical context">
        <div>
          <span>Benchmark ξ</span>
          <strong>{context.benchmark}</strong>
        </div>
        <div>
          <span>Evidence</span>
          <strong>{context.evidence}</strong>
        </div>
        <div>
          <span>Base currency</span>
          <strong>{context.baseCurrency}</strong>
        </div>
        <div>
          <span>As of · Central Time</span>
          <strong>
            {context.asOf
              ? new Date(context.asOf).toLocaleDateString("en-US", { timeZone: "America/Mexico_City" })
              : "Unavailable"}
          </strong>
        </div>
        <div>
          <span>Artifact source</span>
          <strong>{context.source}</strong>
        </div>
      </section>

      <aside className={`sidebar ${mobileOpen ? "mobile-open" : ""}`} aria-label="Primary navigation">
        <div className="sidebar-head">
          <span>Research workspaces</span>
          <button
            className="icon-button desktop-collapse"
            type="button"
            aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
            onClick={() => setCollapsed((value) => !value)}
          >
            <ChevronLeft size={18} />
          </button>
          <button
            className="icon-button mobile-close"
            type="button"
            aria-label="Close navigation"
            onClick={() => setMobileOpen(false)}
          >
            <X size={20} />
          </button>
        </div>
        <nav>
          {workspaceGroups.map((group) => (
            <div className="nav-group" key={group}>
              <span className="nav-group-label">{group}</span>
              {workspaces
                .filter((workspace) => workspace.group === group)
                .map((workspace) => {
                  const href = workspace.slug ? `/${workspace.slug}` : "/";
                  const active = pathname === href;
                  const Icon = workspace.icon;
                  return (
                    <Link
                      className={active ? "active" : ""}
                      href={href}
                      key={workspace.label}
                      onClick={() => setMobileOpen(false)}
                      title={collapsed ? workspace.label : undefined}
                    >
                      <Icon size={18} aria-hidden="true" />
                      <span>{workspace.label}</span>
                    </Link>
                  );
                })}
            </div>
          ))}
        </nav>
        <footer>
          <span>Public-data grade</span>
          <strong>Paper execution only</strong>
        </footer>
      </aside>
      {mobileOpen ? <button className="nav-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} /> : null}
      <main id="main-content">{children}</main>
    </div>
  );
}
