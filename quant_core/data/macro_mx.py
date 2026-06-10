"""Mexican macro and rates providers (Banxico SIE + INEGI BIE), zero cost.

Both APIs are free with registered tokens:
- Banxico SIE token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
  (env ``BANXICO_TOKEN``)
- INEGI BIE token: https://www.inegi.org.mx/app/api/denue/v1/tokenVerify.aspx
  (env ``INEGI_TOKEN``)

Without a token each provider degrades to an empty frame (honest degradation:
the freshness report shows the gap instead of fabricating data).
"""

from __future__ import annotations

import os
import urllib.parse

import numpy as np
import pandas as pd

from quant_core.data.base import PERSISTENT_CACHE, http_read_json

# Catalog of high-value SIE series (code -> column name). Sources: Banxico
# SIE catalog; all daily/monthly public series.
BANXICO_SIE_SERIES: dict[str, str] = {
    "SF61745": "POLICY_RATE",  # Tasa objetivo
    "SF60648": "TIIE_28",
    "SF60649": "TIIE_91",
    "SF60650": "TIIE_182",
    "SF45470": "CETES_28",
    "SF45471": "CETES_91",
    "SF45472": "CETES_182",
    "SF45473": "CETES_364",
    "SF43718": "USDMXN_FIX",
    "SP68257": "UDIS",
    "SP1": "INPC",
}

INEGI_BIE_URL = (
    "https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/"
    "{indicators}/es/0700/false/BIE/2.0/{token}?type=json"
)

# Common BIE indicator ids (id -> column name).
INEGI_BIE_SERIES: dict[str, str] = {
    "737121": "IGAE",
    "628194": "INFLACION_QUINCENAL",
    "444612": "TASA_DESOCUPACION",
}


def _to_float(value) -> float:
    try:
        out = float(str(value).replace(",", ""))
        return out if np.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fetch_banxico_series(
    series_map: dict[str, str],
    start,
    end,
    *,
    token: str | None = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch a catalog of Banxico SIE series into one daily frame."""
    token = (token if token is not None else os.getenv("BANXICO_TOKEN", "")).strip()
    if not token or not series_map:
        return pd.DataFrame()
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    payload = {"codes": sorted(series_map), "start": start_s, "end": end_s, "source": "banxico_sie_v2"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("macro_banxico_sie", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    frames = []
    for code, name in series_map.items():
        url = (
            f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{code}/datos/{start_s}/{end_s}"
            f"?token={urllib.parse.quote(token)}"
        )
        try:
            data = http_read_json(url, user_agent="QuantStockPicker/1.0", timeout=timeout)
            datos = data.get("bmx", {}).get("series", [{}])[0].get("datos", [])
            rows = [
                {
                    "Date": pd.to_datetime(item.get("fecha"), dayfirst=True, errors="coerce"),
                    name: _to_float(item.get("dato", "")),
                }
                for item in datos
            ]
            frame = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date")
            if not frame.empty:
                frames.append(frame)
        except Exception:
            continue
    out = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if use_cache and not out.empty:
        PERSISTENT_CACHE.set_df("macro_banxico_sie", payload, out)
    return out


def fetch_banxico_rates(start, end) -> pd.DataFrame:
    """Backward-compatible policy-rate fetch (single series)."""
    return fetch_banxico_series({"SF61745": "POLICY_RATE"}, start, end, use_cache=False)


def fetch_banxico_macro_panel(start, end, *, use_cache: bool = True, cache_ttl_hours: int = 24) -> pd.DataFrame:
    """Full Banxico catalog: policy rate, TIIE, CETES curve, FIX, UDIS, INPC."""
    return fetch_banxico_series(BANXICO_SIE_SERIES, start, end, use_cache=use_cache, cache_ttl_hours=cache_ttl_hours)


def fetch_inegi_series(
    indicator_ids: dict[str, str] | None = None,
    *,
    token: str | None = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 24,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch INEGI BIE indicators (IGAE, inflation, unemployment) as a frame."""
    token = (token if token is not None else os.getenv("INEGI_TOKEN", "")).strip()
    series = indicator_ids or INEGI_BIE_SERIES
    if not token or not series:
        return pd.DataFrame()
    payload = {"ids": sorted(series), "source": "inegi_bie_v1"}
    if use_cache:
        cached = PERSISTENT_CACHE.get_df("macro_inegi_bie", payload, cache_ttl_hours)
        if cached is not None and not cached.empty:
            return cached
    url = INEGI_BIE_URL.format(indicators=",".join(series), token=urllib.parse.quote(token))
    try:
        data = http_read_json(url, user_agent="QuantStockPicker/1.0", timeout=timeout)
    except Exception:
        return pd.DataFrame()
    frames = []
    for item in data.get("Series", []) or []:
        indicator = str(item.get("INDICADOR", "")).strip()
        name = series.get(indicator, indicator)
        rows = []
        for obs in item.get("OBSERVATIONS", []) or []:
            period = str(obs.get("TIME_PERIOD", "")).strip()
            # BIE periods come as 2024/03, 2024/02Q or plain years.
            stamp = pd.to_datetime(period.replace("/", "-"), errors="coerce")
            rows.append({"Date": stamp, name: _to_float(obs.get("OBS_VALUE"))})
        frame = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date")
        if not frame.empty:
            frames.append(frame)
    out = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    if use_cache and not out.empty:
        PERSISTENT_CACHE.set_df("macro_inegi_bie", payload, out)
    return out
