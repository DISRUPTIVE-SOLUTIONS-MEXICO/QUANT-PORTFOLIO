from __future__ import annotations

import argparse
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".quant_cache",
    ".quant_cache_vendor_yf",
    ".ruff_cache",
    "__pycache__",
    "build",
    "data_cache",
    "dist",
    "node_modules",
    "paper",
    "venv",
    ".venv",
}

EXCLUDED_FILES = {
    ".env",
    "audit.jsonl",
    "credentials.toml",
    "secrets.toml",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yml",
    ".yaml",
}

JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")
LONG_TOKEN_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|service[_-]?role[_-]?key)\b\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{32,})"
)
PLACEHOLDER_WORDS = ("replace", "placeholder", "example", "your-", "your_", "optional", "dummy", "test")
PROVIDER_TERMS = ("cl" + "aude", "co" + "dex")
PROVIDER_RE = re.compile("|".join(re.escape(term) for term in PROVIDER_TERMS), re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    rule: str
    detail: str


def _is_placeholder(value: str) -> bool:
    lower = value.lower()
    return any(word in lower for word in PLACEHOLDER_WORDS) or set(value) <= {"x", "X", "*"}


def _decode_jwt_payload(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8"))
    except Exception:
        return {}


def _iter_candidate_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(root).parts)
        if rel_parts & EXCLUDED_DIRS:
            continue
        if path.name in EXCLUDED_FILES or path.name.startswith(".env."):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", ".dockerignore"}:
            continue
        yield path


def scan(root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_candidate_files(root):
        rel = str(path.relative_to(root))
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            provider_match = PROVIDER_RE.search(line)
            if provider_match:
                findings.append(
                    Finding(rel, line_no, "provider_reference", "vendor-specific AI tool reference in public file")
                )
            for token in JWT_RE.findall(line):
                payload = _decode_jwt_payload(token)
                if payload.get("role") == "service_role" or payload.get("iss") == "supabase":
                    findings.append(Finding(rel, line_no, "supabase_jwt", "Supabase JWT-like credential in public file"))
            assignment_match = LONG_TOKEN_ASSIGNMENT_RE.search(line)
            if assignment_match and not _is_placeholder(assignment_match.group(2)):
                findings.append(
                    Finding(rel, line_no, "secret_assignment", "long credential-like value assigned in public file")
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan public project files for leaked secrets and tooling references.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    findings = scan(args.root.resolve())
    if findings:
        for finding in findings:
            print(f"{finding.file}:{finding.line}: {finding.rule}: {finding.detail}")
        return 1
    print("Security hygiene scan passed: no public secret leaks or vendor-specific tooling references found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
