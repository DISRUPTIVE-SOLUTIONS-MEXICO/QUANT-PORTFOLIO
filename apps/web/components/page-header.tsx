interface PageHeaderProps {
  eyebrow: string;
  title: string;
  description: string;
  scope?: string;
}

export function PageHeader({ eyebrow, title, description, scope = "OOS / active snapshot" }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <span className="scope-badge">{scope}</span>
    </header>
  );
}
