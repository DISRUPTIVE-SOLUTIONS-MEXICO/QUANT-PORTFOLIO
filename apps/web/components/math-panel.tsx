"use client";

import { BlockMath } from "react-katex";

interface MathPanelProps {
  title: string;
  formula: string;
  children: React.ReactNode;
}

export function MathPanel({ title, formula, children }: MathPanelProps) {
  return (
    <section className="math-panel">
      <div>
        <p className="eyebrow">Formal construction</p>
        <h2>{title}</h2>
        <div className="math-copy">{children}</div>
      </div>
      <div className="formula-frame" aria-label={`${title} formula`}>
        <BlockMath math={formula} />
      </div>
    </section>
  );
}
