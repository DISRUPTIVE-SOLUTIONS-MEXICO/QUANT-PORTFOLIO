"""Zero-cost, multi-source data layer for Quant Portfolio-Kaizen.

Design principles:

- Every provider is free: public APIs (FRED, Banxico SIE, central banks),
  free-tier feeds (yfinance, Stooq) or governed scraping of official sources.
- Redundancy over trust: price series are cross-validated across providers
  and discrepancies surface in the data-freshness report instead of silently
  propagating into research.
- Provenance everywhere: each fetch records source, URL, retrieval time and a
  content hash so cached research inputs are reproducible and auditable.
"""

from quant_core.data.base import (
    CACHE_DIR,
    PERSISTENT_CACHE,
    PersistentCache,
    http_read_json,
    http_read_text,
    validated_http_url,
)
from quant_core.data.provenance import Provenance
from quant_core.data.reconcile import reconcile_price_frames

__all__ = [
    "CACHE_DIR",
    "PERSISTENT_CACHE",
    "PersistentCache",
    "Provenance",
    "http_read_json",
    "http_read_text",
    "reconcile_price_frames",
    "validated_http_url",
]
