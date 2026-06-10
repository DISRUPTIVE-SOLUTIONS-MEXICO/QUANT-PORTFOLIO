from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from math import gamma
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0
# Unit convention for every XCDR/XODR composite below: all loss terms that are
# mixed additively (downside deviation, CVaR, tracking error) are expressed in
# annualized units via sqrt(252) scaling of daily quantities. Max drawdown is a
# sample-path fraction and enters unscaled by convention. Daily CVaR losses are
# clipped at zero before annualization so all-gain samples cannot shrink (or
# flip the sign of) a risk denominator.
_ANN_SQRT = float(np.sqrt(TRADING_DAYS))


def annualized_cvar_loss(r: pd.Series, alpha: float = 0.95) -> float:
    """Non-negative historical CVaR loss scaled to annual units (sqrt-time)."""
    return max(_historical_cvar_loss(r, alpha=alpha), 0.0) * _ANN_SQRT


@dataclass(frozen=True)
class StrategyConstitution:
    """Frozen research contract used to limit researcher degrees of freedom."""

    strategy_id: str
    version: str = "v1"
    allowed_features: tuple[str, ...] = ()
    allowed_hyperparameters: dict[str, tuple[float, ...]] = field(default_factory=dict)
    benchmark_set: tuple[str, ...] = ()
    complexity_budget: int = 25
    max_trials: int = 50
    primary_metric: str = "XCDR-v2"
    promotion_gates: tuple[str, ...] = (
        "DXCDR",
        "PBO",
        "WRC",
        "SPA",
        "ICIR",
        "OOS_QLIKE",
        "DD",
        "CVaR",
    )

    def complexity_score(self) -> int:
        return (
            len(self.allowed_features)
            + sum(len(v) for v in self.allowed_hyperparameters.values())
            + len(self.benchmark_set)
            + len(self.promotion_gates)
        )

    def is_within_budget(self) -> bool:
        return self.complexity_score() <= int(self.complexity_budget)

    def to_record(self) -> dict[str, Any]:
        out = asdict(self)
        out["complexity_score"] = self.complexity_score()
        out["within_budget"] = self.is_within_budget()
        return out


@dataclass(frozen=True)
class VarianceModelResult:
    model: str
    params: dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    oos_qlike: float = np.nan
    forecast_variance: tuple[float, ...] = ()
    lower_band: tuple[float, ...] = ()
    upper_band: tuple[float, ...] = ()
    status: str = "research_only"

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UncertaintyState:
    """Single-row state vector for robust control and promotion diagnostics."""

    mu: float = np.nan
    ann_vol: float = np.nan
    volterra_h: float = np.nan
    pelt_regime: str = "unknown"
    rmt_noise_fraction: float = np.nan
    effective_rank: float = np.nan
    kalman_state_confidence: float = np.nan
    crlb_mean: float = np.nan
    fisher_information: float = np.nan
    entropy: float = np.nan
    entropy_normalized: float = np.nan
    crowding: float = np.nan
    uncertainty_score: float = np.nan

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


def normalized_entropy(weights: pd.Series | np.ndarray) -> float:
    w = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    w = w[w > 0]
    if w.empty:
        return np.nan
    w = w / w.sum()
    if len(w) <= 1:
        return 0.0
    h = float(-(w * np.log(w)).sum())
    return float(h / np.log(len(w)))


def fractional_volterra_kernel(hurst: float = 0.10, length: int = 126) -> pd.Series:
    """Causal fractional Volterra kernel K(j) proportional to j^(H-1/2)."""
    h = float(np.clip(hurst, 0.01, 0.99))
    n = int(max(2, length))
    lags = np.arange(1, n + 1, dtype=float)
    kernel = np.power(lags, h - 0.5) / max(gamma(h + 0.5), 1e-12)
    kernel = np.maximum(kernel, 0.0)
    kernel = kernel / kernel.sum() if kernel.sum() > 0 else np.ones(n) / n
    return pd.Series(kernel, index=pd.RangeIndex(1, n + 1), name=f"FV_Kernel_H_{h:.3f}")


def _hurst_from_log_vol(
    log_vol: pd.Series,
    max_lag: int = 25,
    min_lag: int = 1,
    min_obs: int = 100,
) -> dict[str, float | str]:
    """Hurst exponent from the q=2 structure function of a log-volatility path.

    Implements the Gatheral-Jaisson-Rosenbaum (2018, Quantitative Finance)
    scaling regression: m(2, d) = E|log v_{t+d} - log v_t|^2 ~ c * d^{2H}, so
    the slope of log m on log d equals 2H.
    """
    x = pd.Series(log_vol).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    out: dict[str, float | str] = {
        "hurst": np.nan,
        "slope": np.nan,
        "r2": np.nan,
        "n_obs": float(len(x)),
        "status": "insufficient_history",
    }
    max_lag = int(max(max_lag, min_lag + 3))
    if len(x) < max(int(min_obs), 4 * max_lag):
        return out
    lags = np.arange(int(max(min_lag, 1)), max_lag + 1)
    m = np.array([float(np.mean(np.abs(x[lag:] - x[:-lag]) ** 2)) for lag in lags])
    valid = np.isfinite(m) & (m > 0)
    if valid.sum() < 4:
        out["status"] = "degenerate_structure_function"
        return out
    lx = np.log(lags[valid].astype(float))
    ly = np.log(m[valid])
    slope, _intercept = np.polyfit(lx, ly, 1)
    corr = np.corrcoef(lx, ly)[0, 1]
    out.update(
        {
            "hurst": float(np.clip(slope / 2.0, 0.01, 0.99)),
            "slope": float(slope),
            "r2": float(corr**2) if np.isfinite(corr) else np.nan,
            "status": "ok",
        }
    )
    return out


def estimate_hurst_rv(
    returns: pd.Series,
    *,
    rv_window: int = 5,
    max_lag: int = 25,
    min_obs: int = 150,
) -> dict[str, float | str]:
    """Estimate H for the rough/power-law variance kernel from daily returns.

    Daily realized variance is proxied by a rolling sum of squared returns;
    structure-function lags start at ``rv_window`` to avoid the spurious
    smoothness induced by overlapping windows at shorter lags.
    """
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < int(min_obs):
        return {
            "hurst": np.nan,
            "slope": np.nan,
            "r2": np.nan,
            "n_obs": float(len(r)),
            "status": "insufficient_history",
        }
    rv = r.pow(2).rolling(int(max(rv_window, 2))).sum()
    log_vol = 0.5 * np.log(rv.clip(lower=1e-18))
    return _hurst_from_log_vol(log_vol.dropna(), max_lag=max_lag, min_lag=int(max(rv_window, 2)), min_obs=min_obs)


def fractional_volterra_variance(
    returns: pd.Series,
    hurst: float | str = 0.10,
    length: int = 126,
    min_periods: int = 20,
) -> pd.Series:
    """Causal daily variance forecast from lagged squared returns only.

    This is a power-law weighted moving average of squared returns (a
    rough-volatility-inspired kernel), not a stochastic Volterra model. Pass
    ``hurst="estimated"`` to calibrate H from the data with
    :func:`estimate_hurst_rv` (falls back to 0.10 when estimation fails).
    """
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.Series(dtype=float)
    if isinstance(hurst, str):
        if hurst.strip().lower() != "estimated":
            raise ValueError(f"Unsupported hurst spec: {hurst!r}; use a float or 'estimated'.")
        est = estimate_hurst_rv(r)
        h_val = est.get("hurst", np.nan)
        hurst = float(h_val) if isinstance(h_val, (int, float)) and np.isfinite(h_val) else 0.10
    k = fractional_volterra_kernel(float(hurst), length)
    eps2 = r.pow(2)
    vals = []
    for pos in range(len(eps2)):
        start = max(0, pos - len(k))
        hist = eps2.iloc[start:pos]
        if len(hist) < min_periods:
            vals.append(np.nan)
            continue
        kk = k.iloc[: len(hist)].iloc[::-1].to_numpy()
        kk = kk / kk.sum() if kk.sum() > 0 else np.ones(len(hist)) / len(hist)
        vals.append(float(np.dot(hist.to_numpy(), kk)))
    return pd.Series(vals, index=r.index, name="FractionalVolterra_Variance")


def qlike_loss(realized_returns: pd.Series | np.ndarray, forecast_variance: pd.Series | np.ndarray) -> float:
    r = pd.Series(realized_returns, dtype=float)
    h = pd.Series(forecast_variance, dtype=float)
    idx = r.index.intersection(h.index) if isinstance(r.index, pd.Index) and isinstance(h.index, pd.Index) else None
    if idx is not None and len(idx) > 0:
        r = r.reindex(idx)
        h = h.reindex(idx)
    frame = pd.DataFrame({"r": r.to_numpy(), "h": h.to_numpy()}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return np.nan
    var = frame["h"].clip(lower=1e-12)
    return float(np.mean(frame["r"].pow(2) / var + np.log(var)))


def rmt_clean_covariance(returns: pd.DataFrame, annualize: float = 252.0) -> tuple[pd.DataFrame, dict[str, float | str]]:
    clean = pd.DataFrame(returns).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0.0)
    if clean.empty:
        return pd.DataFrame(), {"RMT_Status": "empty"}
    x = clean.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    n_obs, n_assets = x.shape
    sample = np.cov(x, rowvar=False)
    if n_assets == 1:
        cov = np.atleast_2d(sample) * annualize
        return pd.DataFrame(cov, index=clean.columns, columns=clean.columns), {
            "RMT_Status": "single_asset",
            "RMT_Noise_Fraction": 0.0,
            "Effective_Rank": 1.0,
            "MP_Lambda_Plus": np.nan,
        }
    eigval, eigvec = np.linalg.eigh((sample + sample.T) / 2.0)
    eigval = np.clip(eigval, 1e-12, None)
    q = n_assets / max(n_obs, 1)
    sigma2 = float(np.median(eigval))
    lambda_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2
    noisy = eigval <= lambda_plus
    clean_eig = eigval.copy()
    if noisy.any():
        clean_eig[noisy] = float(np.mean(eigval[noisy]))
    cleaned = eigvec @ np.diag(np.clip(clean_eig, 1e-12, None)) @ eigvec.T
    cleaned = (cleaned + cleaned.T) / 2.0
    prob = eigval / eigval.sum()
    eff_rank = float(np.exp(-(prob * np.log(np.clip(prob, 1e-12, None))).sum()))
    meta: dict[str, float | str] = {
        "RMT_Status": "ok",
        "RMT_Q": float(q),
        "MP_Lambda_Plus": float(lambda_plus),
        "RMT_Noise_Fraction": float(noisy.mean()),
        "Effective_Rank": eff_rank,
        "Signal_Modes": int((~noisy).sum()),
    }
    return pd.DataFrame(cleaned * annualize, index=clean.columns, columns=clean.columns), meta


def fisher_crlb_mean(returns: pd.DataFrame | pd.Series) -> tuple[pd.Series, pd.Series]:
    """Fisher information and CRLB for the mean under an i.i.d. Gaussian model.

    Honesty note: with i.i.d. Gaussian returns the Cramer-Rao lower bound for
    the sample mean reduces to the classical standard error of the mean,
    CRLB_mu = sigma^2 / T. The CRLB framing documents *why* this is the right
    observation-noise scale for shrinkage, but the quantity itself is the
    textbook SE — no exotic information geometry is involved.
    """
    x = pd.DataFrame(returns).replace([np.inf, -np.inf], np.nan)
    n = x.count().clip(lower=1)
    var = x.var(ddof=1).replace(0.0, np.nan)
    fisher = n / var
    crlb = var / n
    return fisher.replace([np.inf, -np.inf], np.nan), crlb.replace([np.inf, -np.inf], np.nan)


def robust_alpha_shrinkage(alpha: pd.Series, crlb: pd.Series, floor: float = 1e-12) -> pd.Series:
    a = pd.Series(alpha, dtype=float)
    c = pd.Series(crlb, dtype=float).reindex(a.index).fillna(pd.Series(crlb).median())
    denom = a.abs() + np.sqrt(c.clip(lower=floor))
    shrunk = a * (a.abs() / denom.replace(0.0, np.nan))
    return shrunk.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def causal_kalman_mean_filter(
    returns: pd.Series, process_var: float = 1e-7, obs_var: float | None = None
) -> pd.DataFrame:
    r = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.DataFrame(columns=["Filtered_Mean", "State_Variance", "State_Confidence"])
    # Keep the default strictly prefix-causal: do not estimate observation noise
    # from the full sample because that would let late shocks alter early states.
    obs = float(obs_var) if obs_var is not None else float(max(float(r.iloc[0]) ** 2, 1e-8))
    q = float(max(process_var, 1e-12))
    mean = 0.0
    state_var = obs
    rows = []
    for dt, y in r.items():
        pred_mean = mean
        pred_var = state_var + q
        k_gain = pred_var / max(pred_var + obs, 1e-12)
        mean = pred_mean + k_gain * (float(y) - pred_mean)
        state_var = (1.0 - k_gain) * pred_var
        rows.append(
            {
                "Date": dt,
                "Filtered_Mean": mean,
                "State_Variance": state_var,
                "State_Confidence": float(1.0 / (1.0 + state_var / max(obs, 1e-12))),
            }
        )
    return pd.DataFrame(rows).set_index("Date")


def xcdr_v2_score(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    weights: pd.Series | None = None,
    crlb: pd.Series | None = None,
    turnover: float = 0.0,
    lambda_cvar: float = 1.0,
    lambda_drawdown: float = 1.0,
    lambda_uncertainty: float = 1.0,
    lambda_turnover: float = 0.25,
    lambda_entropy: float = 0.10,
) -> dict[str, float | str]:
    p = pd.Series(portfolio_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    idx = p.index.intersection(b.index)
    p = p.reindex(idx)
    b = b.reindex(idx)
    if len(idx) < 20:
        return {"XCDR_v2": np.nan, "DXCDR": np.nan, "XCDR_v2_Status": "insufficient_history"}
    active = p - b
    active_ann = float(active.mean() * 252.0)
    downside = float(np.sqrt(np.mean(np.minimum(active, 0.0) ** 2)) * np.sqrt(252.0))
    cvar = annualized_cvar_loss(active)
    dd = _max_drawdown_loss(active)
    entropy_n = normalized_entropy(weights) if weights is not None else 1.0
    uncertainty = 0.0
    if crlb is not None:
        c = pd.Series(crlb, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        uncertainty = float(np.sqrt(c.clip(lower=0.0)).mean()) if not c.empty else 0.0
    denom = np.sqrt(downside**2 + lambda_cvar * cvar**2 + lambda_drawdown * dd**2 + lambda_uncertainty * uncertainty**2)
    penalty = lambda_turnover * abs(float(turnover)) + lambda_entropy * max(0.0, 1.0 - float(entropy_n))
    score = active_ann / max(denom, 1e-12) - penalty
    base = active_ann / max(downside, 1e-12)
    return {
        "XCDR_v2": float(score),
        "DXCDR": float(score - base),
        "XCDR_v2_Active_Ann_Return": active_ann,
        "XCDR_v2_Denominator": float(denom),
        "XCDR_v2_CVaR_Loss": float(cvar),
        "XCDR_v2_MaxDD_Loss": float(dd),
        "XCDR_v2_Entropy": float(entropy_n),
        "XCDR_v2_Uncertainty": float(uncertainty),
        "XCDR_v2_Turnover": float(turnover),
        "XCDR_v2_Status": "diagnostic_only",
    }


def upside_downside_diagnostics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    baseline_returns: pd.Series | None = None,
    tolerance: float = 0.05,
) -> dict[str, float | bool | str]:
    """Benchmark-relative upside/downside diagnostics.

    Capture ratios use the benchmark sign as the conditioning event. Downside
    capture below one is desirable because both conditional means are negative.
    """
    p = pd.Series(portfolio_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    idx = p.index.intersection(b.index)
    p = p.reindex(idx)
    b = b.reindex(idx)
    if len(idx) < 5:
        return {
            "Upside_Deviation": np.nan,
            "Downside_Deviation": np.nan,
            "Upside_Capture": np.nan,
            "Downside_Capture": np.nan,
            "Return_Gap_to_Xi": np.nan,
            "Downside_Preservation_Pass": False,
            "Downside_Diagnostics_Status": "insufficient_history",
        }
    upside = p.clip(lower=0.0)
    downside = p.clip(upper=0.0)
    b_up_mask = b > 0
    b_down_mask = b < 0
    upside_capture = np.nan
    downside_capture = np.nan
    if b_up_mask.any() and abs(float(b[b_up_mask].mean())) > 1e-12:
        upside_capture = float(p[b_up_mask].mean() / b[b_up_mask].mean())
    if b_down_mask.any() and abs(float(b[b_down_mask].mean())) > 1e-12:
        downside_capture = float(p[b_down_mask].mean() / b[b_down_mask].mean())
    out: dict[str, float | bool | str] = {
        "Upside_Deviation": float(np.sqrt(np.mean(upside**2)) * np.sqrt(252.0)),
        "Downside_Deviation": float(np.sqrt(np.mean(downside**2)) * np.sqrt(252.0)),
        "Upside_Capture": upside_capture,
        "Downside_Capture": downside_capture,
        "Return_Gap_to_Xi": float(b.mean() * 252.0 - p.mean() * 252.0),
        "Portfolio_MaxDD_Loss": _max_drawdown_loss(p),
        "Portfolio_CVaR_Loss_95": _historical_cvar_loss(p),
        "Downside_Diagnostics_Status": "ok",
    }
    if baseline_returns is None:
        out["Downside_Preservation_Pass"] = bool(pd.notna(downside_capture) and downside_capture < 1.0)
        return out
    base = pd.Series(baseline_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().reindex(idx).fillna(0.0)
    cand_dd = _max_drawdown_loss(p)
    base_dd = _max_drawdown_loss(base)
    cand_cvar = _historical_cvar_loss(p)
    base_cvar = _historical_cvar_loss(base)
    cand_down = float(np.sqrt(np.mean(p.clip(upper=0.0) ** 2)) * np.sqrt(252.0))
    base_down = float(np.sqrt(np.mean(base.clip(upper=0.0) ** 2)) * np.sqrt(252.0))
    out.update(
        {
            "Baseline_MaxDD_Loss": base_dd,
            "Baseline_CVaR_Loss_95": base_cvar,
            "Baseline_Downside_Deviation": base_down,
            "MaxDD_Deterioration": cand_dd - base_dd,
            "CVaR_Deterioration": cand_cvar - base_cvar,
            "Downside_Deterioration": cand_down - base_down,
            "Downside_Preservation_Pass": bool(
                cand_dd <= base_dd * (1.0 + tolerance) + 1e-12
                and cand_cvar <= base_cvar * (1.0 + tolerance) + 1e-12
                and cand_down <= base_down * (1.0 + tolerance) + 1e-12
            ),
        }
    )
    return out


def xcdr_v3_growth_control_score(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    weights: pd.Series | None = None,
    crlb: pd.Series | None = None,
    turnover: float = 0.0,
    lambda_downside: float = 1.0,
    lambda_cvar: float = 1.25,
    lambda_drawdown: float = 0.75,
    lambda_tracking: float = 0.25,
    lambda_uncertainty: float = 1.0,
    lambda_turnover: float = 0.25,
    lambda_entropy: float = 0.10,
    upside_reward: float = 0.20,
    convexity_reward: float = 0.10,
    downside_capture_penalty: float = 0.75,
) -> dict[str, float | bool | str]:
    """Asymmetric benchmark-relative return/downside score.

    XCDR-v3 rewards active return, upside capture and beta convexity, while
    penalizing downside deviation, CVaR, drawdown, tracking error, CRLB,
    turnover and concentration. It is a diagnostic/research objective; it does
    not replace promotion gates.

    Unit convention: downside deviation, tracking error and CVaR all enter the
    risk denominator in annualized units (daily CVaR loss is clipped at zero
    and scaled by sqrt(252)); max drawdown stays a sample-path fraction. The
    nominal lambdas therefore act on commensurate magnitudes. Reported
    ``XCDR_v3_CVaR_Loss`` is the annualized clipped value.
    """
    p = pd.Series(portfolio_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    idx = p.index.intersection(b.index)
    p = p.reindex(idx)
    b = b.reindex(idx)
    if len(idx) < 20:
        return {
            "XCDR_v3_GrowthControl": np.nan,
            "DXCDR_v3": np.nan,
            "XCDR_v3_Status": "insufficient_history",
            "XCDR_v3_Capture_Pass": False,
        }

    active = p - b
    active_ann = float(active.mean() * 252.0)
    downside = float(np.sqrt(np.mean(np.minimum(p, 0.0) ** 2)) * np.sqrt(252.0))
    cvar = annualized_cvar_loss(p)
    dd = _max_drawdown_loss(p)
    te = float(active.std(ddof=1) * np.sqrt(252.0)) if len(active) > 2 else 0.0
    diag = upside_downside_diagnostics(p, b)
    uc = float(diag.get("Upside_Capture", np.nan))
    dc = float(diag.get("Downside_Capture", np.nan))

    up = b > 0
    dn = b < 0
    beta_up = np.nan
    beta_dn = np.nan
    if up.sum() > 3 and float(np.var(b[up], ddof=1)) > 1e-12:
        beta_up = float(np.cov(p[up], b[up])[0, 1] / np.var(b[up], ddof=1))
    if dn.sum() > 3 and float(np.var(b[dn], ddof=1)) > 1e-12:
        beta_dn = float(np.cov(p[dn], b[dn])[0, 1] / np.var(b[dn], ddof=1))
    convexity = float(beta_up - beta_dn) if np.isfinite(beta_up) and np.isfinite(beta_dn) else 0.0

    entropy_n = normalized_entropy(weights) if weights is not None else 1.0
    uncertainty = 0.0
    if crlb is not None:
        c = pd.Series(crlb, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        uncertainty = float(np.sqrt(c.clip(lower=0.0)).mean()) if not c.empty else 0.0

    denom = np.sqrt(
        lambda_downside * downside**2
        + lambda_cvar * cvar**2
        + lambda_drawdown * dd**2
        + lambda_tracking * te**2
        + lambda_uncertainty * uncertainty**2
        + 1e-12
    )
    upside_term = upside_reward * max(0.0, uc - 1.0) if np.isfinite(uc) else 0.0
    convexity_term = convexity_reward * max(0.0, convexity)
    downside_capture_excess = max(0.0, dc - 1.0) if np.isfinite(dc) else 1.0
    upside_capture_shortfall = max(0.0, 1.0 - uc) if np.isfinite(uc) else 1.0
    penalty = (
        downside_capture_penalty * downside_capture_excess
        + 0.35 * upside_capture_shortfall
        + lambda_turnover * abs(float(turnover))
        + lambda_entropy * max(0.0, 1.0 - float(entropy_n))
    )
    score = (active_ann + upside_term + convexity_term) / max(denom, 1e-12) - penalty
    base = active_ann / max(downside, 1e-12)
    capture_pass = bool(np.isfinite(uc) and np.isfinite(dc) and uc > 1.0 and dc < 1.0)
    return {
        "XCDR_v3_GrowthControl": float(score),
        "DXCDR_v3": float(score - base),
        "XCDR_v3_Active_Ann_Return": active_ann,
        "XCDR_v3_Denominator": float(denom),
        "XCDR_v3_Upside_Capture": uc,
        "XCDR_v3_Downside_Capture": dc,
        "XCDR_v3_Beta_Up": float(beta_up) if np.isfinite(beta_up) else np.nan,
        "XCDR_v3_Beta_Down": float(beta_dn) if np.isfinite(beta_dn) else np.nan,
        "XCDR_v3_Convexity_Proxy": float(convexity),
        "XCDR_v3_CVaR_Loss": float(cvar),
        "XCDR_v3_MaxDD_Loss": float(dd),
        "XCDR_v3_Downside_Ann": float(downside),
        "XCDR_v3_Tracking_Error": float(te),
        "XCDR_v3_Entropy": float(entropy_n),
        "XCDR_v3_Uncertainty": float(uncertainty),
        "XCDR_v3_Turnover": float(turnover),
        "XCDR_v3_Capture_Pass": capture_pass,
        "XCDR_v3_Status": "diagnostic_only",
    }


_XCDR_V3_LAMBDA_NAMES = (
    "lambda_downside",
    "lambda_cvar",
    "lambda_drawdown",
    "lambda_tracking",
    "lambda_uncertainty",
    "lambda_turnover",
    "lambda_entropy",
    "upside_reward",
    "convexity_reward",
    "downside_capture_penalty",
)


def xcdr_lambda_sensitivity(
    policy_returns: dict[str, pd.Series] | pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    perturbation: float = 0.50,
    score_kwargs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Sensitivity of the XCDR-v3 policy ranking to its scalarization lambdas.

    Each lambda of :func:`xcdr_v3_growth_control_score` is perturbed by
    ``+/- perturbation`` (multiplicative) and the cross-policy ranking is
    recomputed. The lambdas are researcher degrees of freedom; a ranking that
    flips under modest perturbation means the scalarization, not the data, is
    choosing the winner. One row per (lambda, direction) with the base/new top
    policy and the Spearman correlation between base and perturbed rankings.
    """
    frame = pd.DataFrame(policy_returns)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1)
    if frame.shape[1] < 2:
        return pd.DataFrame()
    base_kwargs = dict(score_kwargs or {})
    defaults = {
        name: param.default
        for name, param in inspect.signature(xcdr_v3_growth_control_score).parameters.items()
        if name in _XCDR_V3_LAMBDA_NAMES
    }

    def _scores(kwargs: dict[str, float]) -> pd.Series:
        vals = {
            col: float(
                xcdr_v3_growth_control_score(frame[col].dropna(), benchmark_returns, **kwargs).get(
                    "XCDR_v3_GrowthControl", np.nan
                )
            )
            for col in frame.columns
        }
        return pd.Series(vals, dtype=float)

    base_scores = _scores(base_kwargs)
    if base_scores.dropna().empty:
        return pd.DataFrame()
    base_rank = base_scores.rank(ascending=False)
    base_top = str(base_scores.idxmax())
    rows = []
    for name in _XCDR_V3_LAMBDA_NAMES:
        current = float(base_kwargs.get(name, defaults[name]))
        for direction, mult in (("down", 1.0 - perturbation), ("up", 1.0 + perturbation)):
            kwargs = dict(base_kwargs)
            kwargs[name] = current * mult
            scores = _scores(kwargs)
            rank = scores.rank(ascending=False)
            spearman = float(base_rank.corr(rank, method="spearman")) if scores.notna().sum() >= 2 else np.nan
            top = str(scores.idxmax()) if scores.notna().any() else ""
            rows.append(
                {
                    "Lambda": name,
                    "Direction": direction,
                    "Multiplier": mult,
                    "Base_Value": current,
                    "Perturbed_Value": current * mult,
                    "Top_Policy_Base": base_top,
                    "Top_Policy_Perturbed": top,
                    "Top_Changed": bool(top != base_top),
                    "Rank_Spearman_vs_Base": spearman,
                    "N_Policies": int(frame.shape[1]),
                }
            )
    out = pd.DataFrame(rows)
    out["Ranking_Fragile"] = out["Top_Changed"] | (out["Rank_Spearman_vs_Base"] < 0.8)
    return out


def xodr_v1_omega_dominance_score(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    omega_returns: pd.DataFrame,
    *,
    weights: pd.Series | None = None,
    crlb: pd.Series | None = None,
    turnover: float = 0.0,
    upside_quantile: float = 0.75,
    downside_quantile: float = 0.25,
    lambda_active: float = 1.00,
    lambda_convexity: float = 0.20,
    lambda_cvar: float = 1.25,
    lambda_drawdown: float = 1.00,
    lambda_uncertainty: float = 1.00,
    lambda_turnover: float = 0.25,
    lambda_entropy: float = 0.10,
    lambda_omega_penalty: float = 4.00,
) -> dict[str, float | bool | str]:
    """Omega-relative upside/downside dominance diagnostic.

    XODR-v1 compares the portfolio against a robust benchmark frontier rather
    than one cherry-picked index. The stress set Omega is used only over the
    supplied sample; callers must pass train/validation/test slices explicitly
    to preserve causality.

    Unit convention: portfolio and Omega CVaR losses are annualized
    (sqrt(252) of the non-negative daily loss) so that frontier breaches and
    the risk denominator combine commensurate magnitudes; max drawdown stays a
    sample-path fraction.
    """
    p = pd.Series(portfolio_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    xi = pd.Series(benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    omega = pd.DataFrame(omega_returns).replace([np.inf, -np.inf], np.nan)
    idx = p.index.intersection(xi.index).intersection(omega.dropna(how="all").index)
    omega = omega.reindex(idx).dropna(axis=1, how="all").fillna(0.0)
    p = p.reindex(idx)
    xi = xi.reindex(idx)
    if len(idx) < 20 or omega.shape[1] < 2:
        return {
            "XODR_v1": np.nan,
            "XODR_v1_Status": "insufficient_history",
            "XODR_v1_Pass": False,
        }

    m_omega = omega.mean(axis=1)
    up_mask = m_omega > 0
    down_mask = m_omega < 0
    if up_mask.sum() < 5 or down_mask.sum() < 5:
        return {
            "XODR_v1": np.nan,
            "XODR_v1_Status": "insufficient_omega_states",
            "XODR_v1_Pass": False,
        }

    omega_up_means = omega.loc[up_mask].mean(axis=0)
    omega_down_means = omega.loc[down_mask].mean(axis=0)
    omega_downside = np.sqrt(np.mean(np.minimum(omega, 0.0) ** 2, axis=0)) * np.sqrt(252.0)
    omega_cvar = omega.apply(annualized_cvar_loss)
    omega_dd = omega.apply(_max_drawdown_loss)

    upside_frontier = float(omega_up_means.quantile(float(np.clip(upside_quantile, 0.5, 0.95))))
    downside_frontier = float(omega_downside.quantile(float(np.clip(downside_quantile, 0.05, 0.5))))
    cvar_frontier = float(omega_cvar.quantile(float(np.clip(downside_quantile, 0.05, 0.5))))
    dd_frontier = float(omega_dd.quantile(float(np.clip(downside_quantile, 0.05, 0.5))))

    p_up_mean = float(p.loc[up_mask].mean())
    p_down_mean = float(p.loc[down_mask].mean())
    omega_down_frontier_mean = float(omega_down_means.quantile(float(np.clip(downside_quantile, 0.05, 0.5))))
    uc_omega = p_up_mean / upside_frontier if abs(upside_frontier) > 1e-12 else np.nan
    dc_omega = p_down_mean / omega_down_frontier_mean if abs(omega_down_frontier_mean) > 1e-12 else np.nan

    active_ann = float((p - xi).mean() * 252.0)
    downside = float(np.sqrt(np.mean(np.minimum(p, 0.0) ** 2)) * np.sqrt(252.0))
    cvar = annualized_cvar_loss(p)
    dd = _max_drawdown_loss(p)

    beta_up = np.nan
    beta_dn = np.nan
    if up_mask.sum() > 3 and float(np.var(m_omega.loc[up_mask], ddof=1)) > 1e-12:
        beta_up = float(np.cov(p.loc[up_mask], m_omega.loc[up_mask])[0, 1] / np.var(m_omega.loc[up_mask], ddof=1))
    if down_mask.sum() > 3 and float(np.var(m_omega.loc[down_mask], ddof=1)) > 1e-12:
        beta_dn = float(np.cov(p.loc[down_mask], m_omega.loc[down_mask])[0, 1] / np.var(m_omega.loc[down_mask], ddof=1))
    convexity = float(beta_up - beta_dn) if np.isfinite(beta_up) and np.isfinite(beta_dn) else 0.0

    entropy_n = normalized_entropy(weights) if weights is not None else 1.0
    uncertainty = 0.0
    if crlb is not None:
        c = pd.Series(crlb, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        uncertainty = float(np.sqrt(c.clip(lower=0.0)).mean()) if not c.empty else 0.0

    omega_penalty = (
        max(0.0, downside - downside_frontier) ** 2
        + lambda_cvar * max(0.0, cvar - cvar_frontier) ** 2
        + lambda_drawdown * max(0.0, dd - dd_frontier) ** 2
    )
    numerator = (
        (max(0.0, uc_omega - 1.0) if np.isfinite(uc_omega) else 0.0)
        + lambda_active * active_ann
        + lambda_convexity * max(0.0, convexity)
    )
    denominator = (
        downside
        + lambda_cvar * cvar
        + lambda_drawdown * dd
        + lambda_uncertainty * uncertainty
        + lambda_turnover * abs(float(turnover))
        + lambda_entropy * max(0.0, 1.0 - float(entropy_n))
        + 1e-12
    )
    score = numerator / denominator - lambda_omega_penalty * omega_penalty
    pass_flag = bool(
        np.isfinite(uc_omega)
        and np.isfinite(dc_omega)
        and uc_omega > 1.0
        and dc_omega < 1.0
        and downside <= downside_frontier + 1e-12
        and cvar <= cvar_frontier + 1e-12
        and dd <= dd_frontier + 1e-12
        and active_ann > 0.0
    )
    return {
        "XODR_v1": float(score),
        "XODR_v1_Active_Ann_Return": active_ann,
        "XODR_v1_Upside_Capture_Omega": float(uc_omega) if np.isfinite(uc_omega) else np.nan,
        "XODR_v1_Downside_Capture_Omega": float(dc_omega) if np.isfinite(dc_omega) else np.nan,
        "XODR_v1_Beta_Up_Omega": float(beta_up) if np.isfinite(beta_up) else np.nan,
        "XODR_v1_Beta_Down_Omega": float(beta_dn) if np.isfinite(beta_dn) else np.nan,
        "XODR_v1_Convexity_Proxy": float(convexity),
        "XODR_v1_Portfolio_Downside": float(downside),
        "XODR_v1_Portfolio_CVaR": float(cvar),
        "XODR_v1_Portfolio_MaxDD": float(dd),
        "XODR_v1_Omega_Upside_Frontier": upside_frontier,
        "XODR_v1_Omega_Downside_Frontier": downside_frontier,
        "XODR_v1_Omega_CVaR_Frontier": cvar_frontier,
        "XODR_v1_Omega_MaxDD_Frontier": dd_frontier,
        "XODR_v1_Omega_Penalty": float(omega_penalty),
        "XODR_v1_Entropy": float(entropy_n),
        "XODR_v1_Uncertainty": float(uncertainty),
        "XODR_v1_Pass": pass_flag,
        "XODR_v1_Status": "diagnostic_only",
    }


def _historical_cvar_loss(r: pd.Series, alpha: float = 0.95) -> float:
    losses = -pd.Series(r).dropna()
    if losses.empty:
        return 0.0
    var = losses.quantile(alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if not tail.empty else float(var)


def _max_drawdown_loss(r: pd.Series) -> float:
    equity = (1.0 + pd.Series(r).fillna(0.0)).cumprod()
    if equity.empty:
        return 0.0
    dd = equity / equity.cummax() - 1.0
    return float(abs(dd.min()))


def build_uncertainty_state(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
    *,
    pelt_regime: str = "unknown",
    volterra_h: float = 0.10,
) -> UncertaintyState:
    r = pd.DataFrame(returns).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if r.empty:
        return UncertaintyState()
    cov, rmt_meta = rmt_clean_covariance(r)
    if weights is None:
        port = r.mean(axis=1)
    else:
        w = pd.Series(weights, dtype=float)
        common = [c for c in r.columns if c in w.index]
        if not common:
            port = r.mean(axis=1)
        else:
            ww = w.reindex(common).fillna(0.0)
            ww = ww / ww.sum() if ww.sum() > 0 else pd.Series(1.0 / len(common), index=common)
            port = r[common].fillna(0.0).dot(ww)
    fisher, crlb = fisher_crlb_mean(r)
    kalman = causal_kalman_mean_filter(port)
    entropy = normalized_entropy(weights) if weights is not None else np.nan
    uncertainty = float(np.nanmean(np.sqrt(crlb.clip(lower=0.0)))) if not crlb.empty else np.nan
    ann_vol = float(port.std(ddof=1) * np.sqrt(252.0)) if len(port) > 1 else np.nan
    return UncertaintyState(
        mu=float(port.mean() * 252.0),
        ann_vol=ann_vol,
        volterra_h=float(volterra_h),
        pelt_regime=str(pelt_regime),
        rmt_noise_fraction=float(rmt_meta.get("RMT_Noise_Fraction", np.nan)),
        effective_rank=float(rmt_meta.get("Effective_Rank", np.nan)),
        kalman_state_confidence=float(kalman["State_Confidence"].iloc[-1]) if not kalman.empty else np.nan,
        crlb_mean=float(crlb.mean()) if not crlb.empty else np.nan,
        fisher_information=float(fisher.mean()) if not fisher.empty else np.nan,
        entropy=entropy,
        entropy_normalized=entropy,
        crowding=float(1.0 / max(float(rmt_meta.get("Effective_Rank", np.nan)), 1e-12))
        if pd.notna(rmt_meta.get("Effective_Rank", np.nan))
        else np.nan,
        uncertainty_score=uncertainty,
    )
