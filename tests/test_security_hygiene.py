from __future__ import annotations

from pathlib import Path

from scripts.security_hygiene_scan import scan


ROOT = Path(__file__).resolve().parents[1]


def test_public_files_do_not_contain_secret_leaks_or_tooling_provenance():
    assert scan(ROOT) == []


def test_sensitive_local_files_are_excluded_from_git_and_docker_contexts():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    required_git_patterns = [
        ".env",
        ".env.*",
        ".streamlit/secrets.toml",
        ".streamlit/credentials.toml",
        ".quant_cache/",
        ".quant_cache_vendor_yf/",
        "audit.jsonl",
        "*.jsonl",
    ]
    required_docker_patterns = [
        ".env",
        ".env.*",
        ".streamlit/secrets.toml",
        ".streamlit/credentials.toml",
        ".quant_cache/",
        ".quant_cache_vendor_yf/",
        "audit.jsonl",
        "*.jsonl",
    ]

    for pattern in required_git_patterns:
        assert pattern in gitignore
    for pattern in required_docker_patterns:
        assert pattern in dockerignore

