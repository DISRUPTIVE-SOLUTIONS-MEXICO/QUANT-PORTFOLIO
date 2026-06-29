from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".quant_cache" / "cloud"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "public_artifacts"
DEFAULT_RESEARCH_DIR = PROJECT_ROOT / "research_artifacts"

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


def _is_nullish(value: Any) -> bool:
    try:
        if value is None:
            return True
        if isinstance(value, float) and not math.isfinite(value):
            return True
        return bool(pd.isna(value))
    except Exception:
        return False


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
    if _is_nullish(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return sanitize_public_artifact(frame.where(pd.notna(frame), None).to_dict("records"))


def _read_research_csv(research_dir: Path, name: str) -> pd.DataFrame:
    path = research_dir / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _select_objective(summary: pd.DataFrame) -> str | None:
    if not isinstance(summary, pd.DataFrame) or summary.empty or "objective" not in summary.columns:
        return None
    preferred = (
        "enhanced_growth_anchor_dd_budget_policy",
        "state_optimized_xcdr_v3_policy",
        "downside_preserving_growth_policy",
    )
    objectives = summary["objective"].dropna().astype(str).tolist()
    for objective in preferred:
        if objective in objectives:
            return objective
    return objectives[0] if objectives else None


def _drawdown_from_return(returns: pd.Series) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    nav = (1.0 + r).cumprod()
    if nav.empty:
        return pd.Series(dtype=float)
    return nav / nav.cummax() - 1.0


def _build_xcdr_strategy_lab(research_dir: Path) -> dict[str, Any]:
    summary = _read_research_csv(research_dir, "xcdr_v3_parallel_research_summary.csv")
    daily_summary = _read_research_csv(research_dir, "xcdr_v3_parallel_research_daily_summary.csv")
    daily = _read_research_csv(research_dir, "xcdr_v3_parallel_research_daily_oos.csv")
    weights = _read_research_csv(research_dir, "xcdr_v3_parallel_research_weights.csv")
    windows = _read_research_csv(research_dir, "xcdr_v3_parallel_research_windows.csv")
    red_team = _read_research_csv(research_dir, "xcdr_v3_parallel_research_red_team.csv")
    report_path = research_dir / "xcdr_v3_parallel_research_report.json"
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}

    objective = _select_objective(summary)
    if objective is None:
        return {}
    selected_summary = summary[summary["objective"].astype(str).eq(objective)].copy()
    selected_daily = (
        daily_summary[daily_summary["objective"].astype(str).eq(objective)].copy()
        if "objective" in daily_summary.columns
        else pd.DataFrame()
    )
    selected_daily_path = (
        daily[daily["objective"].astype(str).eq(objective)].copy() if "objective" in daily.columns else daily.copy()
    )
    selected_weights = (
        weights[weights["objective"].astype(str).eq(objective)].copy()
        if "objective" in weights.columns
        else weights.copy()
    )
    xi = "ξ"
    for frame in (selected_weights, selected_daily_path, selected_summary):
        if "xi" in frame.columns and not frame["xi"].dropna().empty:
            xi = str(frame["xi"].dropna().astype(str).iloc[0])
            break

    price_paths = pd.DataFrame()
    drawdowns = pd.DataFrame()
    if not selected_daily_path.empty and {"date", "portfolio_return", "xi_return"}.issubset(selected_daily_path.columns):
        path = selected_daily_path.copy()
        path["Date"] = pd.to_datetime(path["date"], errors="coerce")
        path = path.dropna(subset=["Date"]).sort_values("Date")
        portfolio_nav = (1.0 + pd.to_numeric(path["portfolio_return"], errors="coerce").fillna(0.0)).cumprod() * 100.0
        xi_nav = (1.0 + pd.to_numeric(path["xi_return"], errors="coerce").fillna(0.0)).cumprod() * 100.0
        price_paths = pd.DataFrame(
            {
                "Date": path["Date"].dt.strftime("%Y-%m-%d"),
                "XCDR/XODR synthetic strategy price": portfolio_nav,
                f"{xi} benchmark observed price": xi_nav,
            }
        )
        drawdowns = pd.DataFrame(
            {
                "Date": path["Date"].dt.strftime("%Y-%m-%d"),
                "XCDR/XODR strategy drawdown": _drawdown_from_return(path["portfolio_return"]),
                f"{xi} benchmark drawdown": _drawdown_from_return(path["xi_return"]),
            }
        )

    observation_days = len(selected_daily_path) if not selected_daily_path.empty else 0
    research_pass = False
    if not selected_summary.empty and "research_gate_pass" in selected_summary.columns:
        research_pass = bool(str(selected_summary["research_gate_pass"].iloc[0]).lower() == "true")
    holdout_pass = False
    if not selected_summary.empty and "holdout_gate_pass" in selected_summary.columns:
        holdout_pass = bool(str(selected_summary["holdout_gate_pass"].iloc[0]).lower() == "true")

    validation_rows = pd.DataFrame(
        [
            {
                "Gate": "Research gate",
                "Observed": research_pass,
                "Threshold": "WRC/SPA/PBO, ICIR and downside preservation",
                "Pass": research_pass,
            },
            {
                "Gate": "Holdout gate",
                "Observed": holdout_pass,
                "Threshold": "Frozen final holdout",
                "Pass": holdout_pass,
            },
            {
                "Gate": "Minimum windows",
                "Observed": selected_summary["windows"].iloc[0] if "windows" in selected_summary.columns and not selected_summary.empty else None,
                "Threshold": selected_summary["Promotion_Min_Windows"].iloc[0] if "Promotion_Min_Windows" in selected_summary.columns and not selected_summary.empty else 12,
                "Pass": research_pass,
            },
        ]
    )

    return {
        "generation": "public_seed_repo_xcdr_v3",
        "status": "PROMOTED_RESEARCH" if research_pass else "RESEARCH_ONLY",
        "benchmark_xi": xi,
        "observation_days": observation_days,
        "frozen_candidate": objective,
        "summary": _records(summary),
        "daily_summary": _records(daily_summary),
        "oos_summary": _records(selected_daily if not selected_daily.empty else selected_summary),
        "oos_price_paths": _records(price_paths),
        "oos_drawdowns": _records(drawdowns),
        "walk_forward_windows": _records(windows),
        "weights": _records(selected_weights),
        "validation": _records(validation_rows),
        "red_team": _records(red_team),
        "report": sanitize_public_artifact(report),
    }


def _strategy_lab_has_evidence(strategy_lab: Any) -> bool:
    if not isinstance(strategy_lab, dict):
        return False
    for key in ("oos_price_paths", "oos_drawdowns", "weights", "validation", "summary"):
        value = strategy_lab.get(key)
        if isinstance(value, list) and len(value) > 0:
            return True
        if isinstance(value, dict) and bool(value):
            return True
    return False


def _inject_xcdr_research_if_missing(artifact: dict[str, Any], research_dir: Path) -> dict[str, Any]:
    payload = artifact.get("dashboard_payload")
    if not isinstance(payload, dict):
        return artifact
    existing_strategy = payload.get("strategy_lab", {})
    if _strategy_lab_has_evidence(existing_strategy):
        return artifact
    strategy_lab = _build_xcdr_strategy_lab(research_dir)
    if not strategy_lab:
        return artifact

    payload["strategy_lab"] = strategy_lab
    allocation = payload.get("allocation")
    if not isinstance(allocation, dict):
        allocation = {}
    if not allocation.get("recommended_portfolio"):
        allocation["recommended_portfolio"] = strategy_lab.get("weights", [])
    if not allocation.get("weights"):
        allocation["weights"] = strategy_lab.get("weights", [])
    payload["allocation"] = allocation

    charts = payload.get("charts")
    if not isinstance(charts, dict):
        charts = {}
    if not charts.get("price_paths"):
        charts["price_paths"] = strategy_lab.get("oos_price_paths", [])
    if not charts.get("drawdowns"):
        charts["drawdowns"] = strategy_lab.get("oos_drawdowns", [])
    payload["charts"] = charts

    research = payload.get("research")
    if not isinstance(research, dict):
        research = {}
    if not research.get("optimization_grid"):
        research["optimization_grid"] = strategy_lab.get("summary", [])
    if not research.get("overfit_diagnostics"):
        research["overfit_diagnostics"] = strategy_lab.get("validation", [])
    payload["research"] = research

    status = payload.get("status")
    if isinstance(status, dict):
        status.setdefault(
            "promotion",
            [
                {
                    "Status": strategy_lab.get("status", "RESEARCH_ONLY"),
                    "Objective": strategy_lab.get("frozen_candidate"),
                    "Benchmark_Xi": strategy_lab.get("benchmark_xi"),
                    "Reason": "Repository XCDR research artifact injected into public seed fallback.",
                }
            ],
        )
    return artifact


def _stamp_public_seed(artifact: dict[str, Any], *, scope: str, research_dir: Path) -> dict[str, Any]:
    clean = sanitize_public_artifact(artifact)
    if not isinstance(clean, dict):
        return {}
    clean = _inject_xcdr_research_if_missing(clean, research_dir)
    clean["scope"] = scope
    clean["public_seed"] = True
    clean["seed_created_at"] = datetime.now(UTC).isoformat()
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


def write_seed(source_path: Path, target_path: Path, *, scope: str, research_dir: Path) -> int:
    artifact = _read_json(source_path)
    if not artifact:
        raise FileNotFoundError(f"Missing source artifact: {source_path}")
    clean = _stamp_public_seed(artifact, scope=scope, research_dir=research_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with gzip.open(target_path, "wb", compresslevel=9) as fh:
        fh.write(encoded)
    return target_path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Build sanitized public dashboard seed artifacts.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
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
        size = write_seed(source, target, scope=scope, research_dir=args.research_dir)
        print(f"{scope}: {target} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
