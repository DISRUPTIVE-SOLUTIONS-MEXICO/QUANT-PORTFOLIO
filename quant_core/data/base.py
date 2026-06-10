"""Shared cache and validated HTTP plumbing for all data providers.

This module owns the deterministic on-disk cache (`PersistentCache`) and the
scheme/host-validated HTTP readers previously defined in the monolith. The
monolith re-imports these names so existing call sites and test monkeypatches
keep working unchanged.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

# Repo root: quant_core/data/base.py -> quant_core/data -> quant_core -> root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO_ROOT / ".quant_cache"


class PersistentCache:
    """Hash-keyed parquet/json cache with TTL, shared by every provider."""

    def __init__(self, root: Path = CACHE_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, namespace: str, payload: dict) -> tuple[Path, Path]:
        serial = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serial.encode("utf-8")).hexdigest()[:24]
        folder = self.root / namespace
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.parquet", folder / f"{digest}.json"

    def get_df(self, namespace: str, payload: dict, ttl_hours: int) -> pd.DataFrame | None:
        path, meta_path = self._paths(namespace, payload)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            age_hours = (time.time() - float(meta.get("created_at", 0.0))) / 3600.0
            if ttl_hours > 0 and age_hours > ttl_hours:
                return None
            df = pd.read_parquet(path)
            if "__index__" in df.columns:
                df = df.set_index("__index__")
                try:
                    converted = pd.to_datetime(df.index)
                    if converted.notna().any():
                        df.index = converted
                except Exception:
                    pass
            return df
        except Exception:
            return None

    def set_df(self, namespace: str, payload: dict, df: pd.DataFrame) -> None:
        if df is None:
            return
        path, meta_path = self._paths(namespace, payload)
        try:
            out = df.copy()
            if out.index.name is not None or not isinstance(out.index, pd.RangeIndex):
                out = out.reset_index(names="__index__")
            out.to_parquet(path, index=False)
            meta_path.write_text(
                json.dumps(
                    {
                        "namespace": namespace,
                        "payload": payload,
                        "rows": int(len(df)),
                        "columns": list(map(str, df.columns)),
                        "created_at": time.time(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            return

    def inventory(self) -> pd.DataFrame:
        rows = []
        for meta_path in self.root.glob("*/*.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "Namespace": meta.get("namespace"),
                        "Rows": meta.get("rows"),
                        "Created_At": pd.to_datetime(meta.get("created_at"), unit="s"),
                        "Age_Hours": (time.time() - float(meta.get("created_at", 0.0))) / 3600.0,
                        "Key": meta_path.stem,
                    }
                )
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values("Created_At", ascending=False) if rows else pd.DataFrame()


PERSISTENT_CACHE = PersistentCache()


def validated_http_url(url: str) -> str:
    """Allow only http(s) URLs with a hostname; reject file:// and friends."""
    parsed = urllib.parse.urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported public-data URL scheme or host: {parsed.scheme or '<missing>'}")
    return urllib.parse.urlunparse(parsed)


def http_read_json(url: str, user_agent: str = "QuantStockPicker/1.0", timeout: int = 20) -> dict:
    url = validated_http_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})  # noqa: S310
    # URL scheme and host are validated above.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def http_read_text(url: str, user_agent: str = "QuantStockPicker/1.0", timeout: int = 20) -> str:
    url = validated_http_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})  # noqa: S310
    # URL scheme and host are validated above.
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")
