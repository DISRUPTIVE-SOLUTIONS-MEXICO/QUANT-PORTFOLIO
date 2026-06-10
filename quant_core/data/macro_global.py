"""Global macro/rates providers from free official endpoints.

Moved here from the monolith (FRED public CSV, BCB SGS, ECB SDW, Bank of
Canada Valet) plus new zero-cost fallbacks: Bank of England IADB, SNB data
portal, World Bank API and IMF SDMX. The monolith re-exports the moved names
so existing call sites keep working.
"""

from __future__ import annotations

import io
import re
import urllib.parse

import numpy as np
import pandas as pd

from quant_core.data.base import http_read_json, http_read_text


def _to_float(value) -> float:
    try:
        out = float(str(value).replace(",", ""))
        return out if np.isfinite(out) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fetch_fred_series_frame(code: str, start, end, timeout: int = 12) -> pd.DataFrame:
    """Fetch one FRED series from its public CSV endpoint with a bounded timeout."""
    code = str(code).strip()
    if not code or not re.fullmatch(r"[A-Za-z0-9_-]+", code):
        raise ValueError(f"Invalid FRED series code: {code!r}")
    params = urllib.parse.urlencode(
        {
            "id": code,
            "cosd": pd.Timestamp(start).strftime("%Y-%m-%d"),
            "coed": pd.Timestamp(end).strftime("%Y-%m-%d"),
        }
    )
    text = http_read_text(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}",
        user_agent="QuantPortfolioKaizen/0.2 research@localhost",
        timeout=max(1, int(timeout)),
    )
    raw = pd.read_csv(io.StringIO(text), na_values=[".", ""])
    if raw.empty or len(raw.columns) < 2:
        return pd.DataFrame(columns=[code])
    date_col = raw.columns[0]
    value_col = code if code in raw.columns else raw.columns[-1]
    dates = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce")
    out = pd.DataFrame({code: values.to_numpy()}, index=dates)
    out.index.name = "Date"
    return out.loc[out.index.notna()].sort_index()


def fetch_bcb_sgs_rates(start, end) -> pd.DataFrame:
    """Brazil central bank (BCB SGS) policy rate, public JSON API."""
    series = {432: "POLICY_RATE"}
    frames = []
    start_s = pd.Timestamp(start).strftime("%d/%m/%Y")
    end_s = pd.Timestamp(end).strftime("%d/%m/%Y")
    for code, name in series.items():
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json"
            f"&dataInicial={urllib.parse.quote(start_s)}&dataFinal={urllib.parse.quote(end_s)}"
        )
        try:
            data = http_read_json(url, timeout=30)
            df = pd.DataFrame(data)
            if df.empty:
                continue
            df["Date"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            df[name] = pd.to_numeric(df["valor"].str.replace(",", ".", regex=False), errors="coerce")
            frames.append(df[["Date", name]].dropna(subset=["Date"]).set_index("Date"))
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_ecb_yield_curve(start, end) -> pd.DataFrame:
    """ECB SDW euro-area government yield curve (3M/2Y/10Y), public CSV API."""
    codes = {
        "SR_3M": "POLICY_RATE",
        "SR_2Y": "SOV_2Y",
        "SR_10Y": "SOV_10Y",
    }
    frames = []
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    for maturity, name in codes.items():
        url = (
            "https://data-api.ecb.europa.eu/service/data/YC/"
            f"B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity}?startPeriod={start_s}&endPeriod={end_s}&format=csvdata"
        )
        try:
            df = pd.read_csv(url)
            if df.empty or "TIME_PERIOD" not in df or "OBS_VALUE" not in df:
                continue
            df["Date"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
            df[name] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
            frames.append(df[["Date", name]].dropna(subset=["Date"]).set_index("Date"))
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_bank_of_canada_rates(start, end) -> pd.DataFrame:
    """Bank of Canada Valet API: policy rate and 2Y/10Y benchmark yields."""
    codes = {
        "V39079": "POLICY_RATE",
        "BD.CDN.2YR.DQ.YLD": "SOV_2Y",
        "BD.CDN.10YR.DQ.YLD": "SOV_10Y",
    }
    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")
    frames = []
    for code, name in codes.items():
        url = f"https://www.bankofcanada.ca/valet/observations/{urllib.parse.quote(code)}/json?start_date={start_s}&end_date={end_s}"
        try:
            data = http_read_json(url, timeout=30)
            rows = []
            for obs in data.get("observations", []):
                rows.append(
                    {"Date": pd.to_datetime(obs.get("d"), errors="coerce"), name: _to_float(obs.get(code, {}).get("v"))}
                )
            df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date")
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def fetch_boe_iadb_series(series_codes: dict[str, str], start, end, timeout: int = 30) -> pd.DataFrame:
    """Bank of England IADB CSV endpoint (e.g. IUDBEDR = Bank Rate)."""
    if not series_codes:
        return pd.DataFrame()
    start_s = pd.Timestamp(start).strftime("%d/%b/%Y")
    end_s = pd.Timestamp(end).strftime("%d/%b/%Y")
    url = (
        "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes"
        f"&Datefrom={urllib.parse.quote(start_s)}&Dateto={urllib.parse.quote(end_s)}"
        f"&SeriesCodes={','.join(series_codes)}&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
    )
    try:
        text = http_read_text(url, user_agent="QuantStockPicker/1.0", timeout=timeout)
        raw = pd.read_csv(io.StringIO(text))
    except Exception:
        return pd.DataFrame()
    if raw.empty or "DATE" not in {c.upper() for c in raw.columns}:
        return pd.DataFrame()
    date_col = next(c for c in raw.columns if c.upper() == "DATE")
    out = pd.DataFrame(index=pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce"))
    for code, name in series_codes.items():
        if code in raw.columns:
            out[name] = pd.to_numeric(raw[code].to_numpy(), errors="coerce")
    out.index.name = "Date"
    return out.loc[out.index.notna()].sort_index()


def fetch_snb_series(cube: str, start, end, timeout: int = 30) -> pd.DataFrame:
    """Swiss National Bank data portal CSV (e.g. cube ``snboffzisa`` for policy rate)."""
    url = f"https://data.snb.ch/api/cube/{urllib.parse.quote(cube)}/data/csv/en"
    try:
        text = http_read_text(url, user_agent="QuantStockPicker/1.0", timeout=timeout)
    except Exception:
        return pd.DataFrame()
    # SNB CSVs carry a metadata preamble; the data block starts at the
    # header line containing "Date".
    lines = text.splitlines()
    start_idx = next((i for i, line in enumerate(lines) if line.lower().startswith("date")), None)
    if start_idx is None:
        return pd.DataFrame()
    try:
        raw = pd.read_csv(io.StringIO("\n".join(lines[start_idx:])), sep=";")
    except Exception:
        return pd.DataFrame()
    if raw.empty or "Date" not in raw.columns or "Value" not in raw.columns:
        return pd.DataFrame()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    out = raw.dropna(subset=["Date"]).set_index("Date")[["Value"]].rename(columns={"Value": cube.upper()})
    mask = (out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))
    return out.loc[mask].sort_index()


def fetch_worldbank_indicator(
    country_iso3: str,
    indicator: str,
    start_year: int,
    end_year: int,
    timeout: int = 30,
) -> pd.DataFrame:
    """World Bank open API (annual macro fallback, no key required)."""
    url = (
        f"https://api.worldbank.org/v2/country/{urllib.parse.quote(country_iso3)}/indicator/"
        f"{urllib.parse.quote(indicator)}?format=json&date={int(start_year)}:{int(end_year)}&per_page=200"
    )
    try:
        data = http_read_json(url, timeout=timeout)
    except Exception:
        return pd.DataFrame()
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return pd.DataFrame()
    rows = [
        {"Date": pd.to_datetime(str(item.get("date")), errors="coerce"), indicator: _to_float(item.get("value"))}
        for item in data[1]
    ]
    out = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    return out


def fetch_imf_sdmx_series(
    dataset: str,
    key: str,
    start_year: int,
    end_year: int,
    timeout: int = 30,
) -> pd.DataFrame:
    """IMF SDMX JSON CompactData endpoint (quarterly/monthly macro, no key)."""
    url = (
        f"https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/{urllib.parse.quote(dataset)}/"
        f"{urllib.parse.quote(key)}?startPeriod={int(start_year)}&endPeriod={int(end_year)}"
    )
    try:
        data = http_read_json(url, timeout=timeout)
    except Exception:
        return pd.DataFrame()
    series = (((data.get("CompactData") or {}).get("DataSet") or {}).get("Series")) or {}
    if isinstance(series, list):
        series = series[0] if series else {}
    obs = series.get("Obs") or []
    if isinstance(obs, dict):
        obs = [obs]
    rows = []
    for item in obs:
        period = str(item.get("@TIME_PERIOD", "")).strip()
        stamp = pd.PeriodIndex([period], freq="Q").to_timestamp(how="end") if "Q" in period.upper() else None
        try:
            date = stamp[0] if stamp is not None else pd.to_datetime(period, errors="coerce")
        except Exception:
            date = pd.to_datetime(period, errors="coerce")
        rows.append({"Date": date, f"{dataset}:{key}": _to_float(item.get("@OBS_VALUE"))})
    out = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    return out
