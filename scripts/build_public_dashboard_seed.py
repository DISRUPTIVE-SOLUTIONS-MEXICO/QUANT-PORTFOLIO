from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".quant_cache" / "cloud"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "public_artifacts"

BLOCKED_KEY_FRAGMENTS = (
    "side_sleeve",
    "side_boom",
    "private_side",
    "mnpi",
)
BLOCKED_EXACT_KEYS = {
    "side_pelt_regime_segments",
    "side_pelt_change_points",
    "side_pelt_timeline",
}
PRIVATE_LABEL_REPLACEMENTS = {
    "Private Side Alpha": "Research strategy",
    "private side alpha": "research strategy",
    "Side Boom": "Research strategy",
    "side boom": "research strategy",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_string(value: str) -> str:
    out = value
    for old, new in PRIVATE_LABEL_REPLACEMENTS.items():
        out = out.replace(old, new)
    return out


def _is_blocked_key(key: str) -> bool:
    low = key.lower()
    if key in BLOCKED_EXACT_KEYS:
        return True
    return any(fragment in low for fragment in BLOCKED_KEY_FRAGMENTS)


def sanitize_public_artifact(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            if _is_blocked_key(str(key)):
                continue
            clean[key] = sanitize_public_artifact(child)
        return clean
    if isinstance(value, list):
        return [sanitize_public_artifact(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def _stamp_public_seed(artifact: dict[str, Any], *, scope: str) -> dict[str, Any]:
    clean = sanitize_public_artifact(artifact)
    if not isinstance(clean, dict):
        return {}
    clean["scope"] = scope
    clean["public_seed"] = True
    clean["seed_created_at"] = datetime.now(timezone.utc).isoformat()
    payload = clean.get("dashboard_payload")
    if isinstance(payload, dict):
        contract = payload.get("contract")
        if not isinstance(contract, dict):
            contract = {}
        contract.update(
            {
                "public_seed": True,
                "seed_scope": scope,
                "seed_disclaimer": (
                    "Sanitized public-data dashboard seed. Supabase artifacts remain the production source of truth."
                ),
            }
        )
        payload["contract"] = contract
        allocation = payload.get("allocation")
        if isinstance(allocation, dict):
            allocation.pop("side_sleeve", None)
        research = payload.get("research")
        if isinstance(research, dict):
            for key in list(research):
                if _is_blocked_key(str(key)):
                    research.pop(key, None)
    return clean


def write_seed(source_path: Path, target_path: Path, *, scope: str) -> int:
    artifact = _read_json(source_path)
    if not artifact:
        raise FileNotFoundError(f"Missing source artifact: {source_path}")
    clean = _stamp_public_seed(artifact, scope=scope)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with gzip.open(target_path, "wb", compresslevel=9) as fh:
        fh.write(encoded)
    return target_path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sanitized public dashboard seed artifacts.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    args = parser.parse_args()

    outputs = {
        "full_analysis": (
            args.source_dir / "latest_full_analysis_payload.json",
            args.target_dir / "latest_full_dashboard_payload.seed.json.gz",
        ),
        "daily_snapshot": (
            args.source_dir / "latest_daily_snapshot_payload.json",
            args.target_dir / "latest_daily_dashboard_payload.seed.json.gz",
        ),
    }
    for scope, (source, target) in outputs.items():
        size = write_seed(source, target, scope=scope)
        print(f"{scope}: {target} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
