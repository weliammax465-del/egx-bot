"""
filters.py
----------
Pre-scoring exclusion filters for the EGX Bot Liquidity-First Strategy v2.

These gates run AFTER data validation but BEFORE indicator computation.
A stock that fails any gate is excluded entirely — no indicators, no score.

Gates implemented:
  1. Liquidity Gate — 20-day avg turnover must be >= MIN_TURNOVER_EGP
  2. Price Limit Gate — |daily change| must be < PRICE_LIMIT_THRESHOLD_PCT

All thresholds come from config.py — no magic numbers here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of a pre-scoring gate check."""
    passed: bool
    gate_name: str           # "liquidity", "price_limit", or "all"
    reason: str              # Arabic explanation
    reason_en: str           # English explanation (for logs)
    details: dict = field(default_factory=dict)


# ─── Individual Gates ────────────────────────────────────────────────────────

def liquidity_gate(
    df: pd.DataFrame,
    ticker: str = "",
    avg_days: int = None,
) -> GateResult:
    """
    Check if a stock meets the minimum liquidity requirement.

    Turnover = Volume × Close price (in EGP).
    Uses the trailing `avg_days` average, not just the latest day,
    to avoid one-day spikes passing the gate.

    Args:
        df: OHLCV DataFrame with 'Volume' and 'Close' columns.
        ticker: Stock symbol (for logging).
        avg_days: Override for the averaging window (default: config.MIN_TURNOVER_AVG_DAYS).

    Returns:
        GateResult with passed=True if turnover >= MIN_TURNOVER_EGP.
    """
    if avg_days is None:
        avg_days = config.MIN_TURNOVER_AVG_DAYS

    # Need enough data for a meaningful average
    if df is None or len(df) < avg_days:
        return GateResult(
            passed=False,
            gate_name="liquidity",
            reason=f"بيانات غير كافية لحساب متوسط السيولة (متاح {len(df) if df is not None else 0} يوم، مطلوب {avg_days})",
            reason_en=f"insufficient data for liquidity avg ({len(df) if df is not None else 0} < {avg_days} days)",
            details={"available_days": len(df) if df is not None else 0, "required_days": avg_days},
        )

    # Compute daily turnover = Volume × Close
    turnover_series = (df["Volume"] * df["Close"]).tail(avg_days)
    avg_turnover = float(turnover_series.mean())
    latest_turnover = float((df["Volume"].iloc[-1] * df["Close"].iloc[-1]))

    passed = avg_turnover >= config.MIN_TURNOVER_EGP

    if passed:
        return GateResult(
            passed=True,
            gate_name="liquidity",
            reason=f"سيولة كافية (متوسط {avg_turnover/1e6:.2f}M EGP/يوم)",
            reason_en=f"liquidity OK (avg {avg_turnover/1e6:.2f}M EGP/day)",
            details={
                "avg_turnover_egp": round(avg_turnover, 2),
                "latest_turnover_egp": round(latest_turnover, 2),
                "min_required_egp": config.MIN_TURNOVER_EGP,
                "avg_days": avg_days,
            },
        )
    else:
        return GateResult(
            passed=False,
            gate_name="liquidity",
            reason=f"سيولة ضعيفة (متوسط {avg_turnover/1e6:.2f}M EGP/يوم، الحد الأدنى {config.MIN_TURNOVER_EGP/1e6:.1f}M)",
            reason_en=f"low liquidity (avg {avg_turnover/1e6:.2f}M EGP/day < {config.MIN_TURNOVER_EGP/1e6:.1f}M minimum)",
            details={
                "avg_turnover_egp": round(avg_turnover, 2),
                "latest_turnover_egp": round(latest_turnover, 2),
                "min_required_egp": config.MIN_TURNOVER_EGP,
                "avg_days": avg_days,
            },
        )


def price_limit_gate(
    daily_change_pct: float,
    ticker: str = "",
) -> GateResult:
    """
    Check if a stock has hit the EGX daily price limit (circuit breaker).

    EGX enforces ±10% daily price limits. Stocks at the limit are effectively
    frozen — you can't get filled at a good price, so we exclude them.

    Args:
        daily_change_pct: Today's percentage change (e.g. 9.5, -10.2).
        ticker: Stock symbol (for logging).

    Returns:
        GateResult with passed=True if |change| < threshold (stock is tradeable).
    """
    if daily_change_pct is None or not np.isfinite(daily_change_pct):
        return GateResult(
            passed=False,
            gate_name="price_limit",
            reason="نسبة التغير غير متاحة أو غير صالحة",
            reason_en="change_pct is None or non-finite",
            details={"change_pct": daily_change_pct},
        )

    abs_change = abs(float(daily_change_pct))
    passed = abs_change < config.PRICE_LIMIT_THRESHOLD_PCT

    if passed:
        return GateResult(
            passed=True,
            gate_name="price_limit",
            reason=f"التغير {daily_change_pct:+.1f}% (ضمن الحد الطبيعي)",
            reason_en=f"change {daily_change_pct:+.1f}% (within normal range)",
            details={"change_pct": daily_change_pct, "threshold": config.PRICE_LIMIT_THRESHOLD_PCT},
        )
    else:
        direction = "صعودي" if daily_change_pct > 0 else "هبوطي"
        return GateResult(
            passed=False,
            gate_name="price_limit",
            reason=f"وصل لحد التذبذب اليومي {direction} ({daily_change_pct:+.1f}%) — السهم مجمد، لا يمكن التنفيذ",
            reason_en=f"price limit hit ({daily_change_pct:+.1f}%) — stock frozen, not tradeable",
            details={"change_pct": daily_change_pct, "threshold": config.PRICE_LIMIT_THRESHOLD_PCT},
        )


# ─── Combined Gate ───────────────────────────────────────────────────────────

def pass_all_gates(
    df: pd.DataFrame,
    daily_change_pct: float,
    ticker: str = "",
) -> tuple[bool, list[GateResult]]:
    """
    Run all pre-scoring gates. Returns (all_passed, results_list).

    A stock must pass ALL gates to proceed to indicator computation.
    The first failing gate stops further checks (fail fast).

    Args:
        df: OHLCV DataFrame with 'Volume' and 'Close' columns.
        daily_change_pct: Today's percentage change.
        ticker: Stock symbol (for logging).

    Returns:
        Tuple of (all_passed, list of GateResult objects for diagnostics).
    """
    results: list[GateResult] = []

    # Gate 1: Liquidity
    liq = liquidity_gate(df, ticker)
    results.append(liq)
    if not liq.passed:
        logger.info(f"  🚫 {ticker}: EXCLUDED — {liq.reason_en}")
        return False, results

    # Gate 2: Price Limit
    pl = price_limit_gate(daily_change_pct, ticker)
    results.append(pl)
    if not pl.passed:
        logger.info(f"  🚫 {ticker}: EXCLUDED — {pl.reason_en}")
        return False, results

    logger.debug(f"  ✅ {ticker}: passed all gates (liquidity + price limit)")
    return True, results


def get_exclusion_reason(results: list[GateResult]) -> str:
    """Extract the Arabic reason from the first failing gate."""
    for r in results:
        if not r.passed:
            return r.reason
    return ""
