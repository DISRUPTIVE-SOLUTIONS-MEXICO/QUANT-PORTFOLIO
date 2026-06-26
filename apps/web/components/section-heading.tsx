interface SectionHeadingProps {
  title: string;
  description: string;
  evidence?: string;
}

export function SectionHeading({ title, description, evidence }: SectionHeadingProps) {
  return (
    <div className="section-heading">
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {evidence ? <span>{evidence}</span> : null}
    </div>
  );
}
