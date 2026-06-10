"""Provenance metadata attached to every external data fetch."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Provenance:
    """Auditable record of where a dataset came from and how it was obtained.

    ``content_hash`` is the sha256 of the raw payload (CSV/JSON/HTML) before
    parsing, so a cached research input can always be traced back to the exact
    bytes the provider served.
    """

    source: str
    url: str = ""
    retrieved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    content_hash: str = ""
    license_note: str = ""
    rows: int = 0
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def content_sha256(raw: str | bytes, length: int = 24) -> str:
    data = raw.encode("utf-8", errors="replace") if isinstance(raw, str) else raw
    return hashlib.sha256(data).hexdigest()[:length]
