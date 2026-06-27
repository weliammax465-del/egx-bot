"""
data/validator.py
-----------------
Data validation layer for EGX market data.

Rejects stale, incomplete, inconsistent, or duplicated market data.
Every piece of data that enters the pipeline must pass through here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Maximum age for data to be considered "fresh" (in hours)
FRESH_THRESHOLD_HOURS = 72  # 3 days — markets are closed on weekends
STALE_THRESHOLD_HOURS = 168  # 7 days — definitely stale

# Minimum number of bars for meaningful technical analysis
MIN_BARS_FOR_ANALYSIS = 50
MIN_BARS_FOR_SMA200 = 200


@dataclass
class ValidationResult:
    """Result of validating a data source."""
    is_valid: bool
    issues: list[str] = field(default_factory=list)
    freshness: str = "unknown"  # "live", "fresh", "stale", "unknown"
    quality_score: float = 1.0  # 0.0 to 1.0
    last_bar_date: Optional[str] = None

    def add_issue(self, msg: str, severity: str = "warning") -> None:
        self.issues.append(f"[{severity}] {msg}")
        if severity == "error":
            self.is_valid = False
            self.quality_score = min(self.quality_score, 0.0)
        elif severity == "warning":
            self.quality_score = min(self.quality_score, self.quality_score * 0.8)


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """
    Validate OHLCV data for a stock.
    Checks: required columns, positive prices, price consistency,
    no NaN/inf, chronological order, minimum length, duplicates, freshness.
    """
    result = ValidationResult(is_valid=True)

    if df is None or len(df) == 0:
        result.add_issue(f"{ticker}: DataFrame is empty", "error")
        return result

    # 1. Check required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.add_issue(f"{ticker}: missing columns {missing}", "error")
        return result

    # 2. Check minimum data length
    if len(df) < MIN_BARS_FOR_ANALYSIS:
        result.add_issue(
            f"{ticker}: insufficient data ({len(df)} bars, need {MIN_BARS_FOR_ANALYSIS})",
            "error"
        )
        return result

    # 3. Check for NaN/inf in critical columns
    for col in ["Open", "High", "Low", "Close"]:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            result.add_issue(f"{ticker}: {nan_count} NaN values in {col}", "warning")

        inf_count = np.isinf(df[col].astype(float)).sum()
        if inf_count > 0:
            result.add_issue(f"{ticker}: {inf_count} inf values in {col}", "error")

    # 4. Check for negative or zero prices
    for col in ["Open", "High", "Low", "Close"]:
        negative = (df[col].astype(float) <= 0).sum()
        if negative > 0:
            result.add_issue(f"{ticker}: {negative} non-positive values in {col}", "error")

    # 5. Check price consistency: High >= Low, High >= Open, High >= Close, etc.
    inconsistent = 0
    for _, row in df.iterrows():
        try:
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            if h < l or h < o or h < c or l > o or l > c:
                inconsistent += 1
        except (TypeError, ValueError):
            inconsistent += 1

    if inconsistent > 0:
        pct = (inconsistent / len(df)) * 100
        if pct > 5:
            result.add_issue(f"{ticker}: {inconsistent} bars with inconsistent OHLC ({pct:.1f}%)", "error")
        else:
            result.add_issue(f"{ticker}: {inconsistent} bars with inconsistent OHLC", "warning")

    # 6. Check for negative volume
    neg_vol = (df["Volume"].astype(float) < 0).sum()
    if neg_vol > 0:
        result.add_issue(f"{ticker}: {neg_vol} bars with negative volume", "error")

    # 7. Check for duplicate dates
    if isinstance(df.index, pd.DatetimeIndex):
        dup_count = df.index.duplicated().sum()
        if dup_count > 0:
            result.add_issue(f"{ticker}: {dup_count} duplicate dates", "warning")

    # 8. Check chronological order
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            result.add_issue(f"{ticker}: dates not in chronological order", "warning")

    # 9. Check data freshness
    result.last_bar_date = str(df.index[-1]) if len(df) > 0 else None
    result.freshness = check_data_freshness(df)

    if result.freshness == "stale":
        result.add_issue(f"{ticker}: data is stale (last bar: {result.last_bar_date})", "warning")
        result.quality_score = min(result.quality_score, 0.5)

    # 10. Check for excessive gaps (missing trading days)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 10:
        # Count unique dates — gaps are normal for weekends/holidays
        # But if we have very few bars relative to the date range, data might be sparse
        date_range = (df.index[-1] - df.index[0]).days
        if date_range > 0:
            bars_per_day = len(df) / date_range
            if bars_per_day < 0.3:  # Less than 0.3 bars per day = very sparse
                result.add_issue(f"{ticker}: sparse data ({len(df)} bars over {date_range} days)", "warning")
                result.quality_score = min(result.quality_score, 0.7)

    return result


def check_data_freshness(df: pd.DataFrame) -> str:
    """
    Check how fresh the data is based on the last bar's date.
    Returns: "live", "fresh", "stale", or "unknown".
    """
    if df is None or len(df) == 0:
        return "unknown"

    if not isinstance(df.index, pd.DatetimeIndex):
        return "unknown"

    last_date = df.index[-1]
    if last_date is None:
        return "unknown"

    # Convert to datetime if needed
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.to_pydatetime()

    now = datetime.now()
    age_hours = (now - last_date).total_seconds() / 3600

    if age_hours < 24:
        return "live"
    elif age_hours < FRESH_THRESHOLD_HOURS:
        return "fresh"
    elif age_hours < STALE_THRESHOLD_HOURS:
        return "stale"
    else:
        return "stale"


def validate_price(price: float, ticker: str = "") -> bool:
    """Validate a single price value."""
    if price is None:
        return False
    try:
        p = float(price)
        return bool(p > 0 and np.isfinite(p))
    except (TypeError, ValueError):
        return False


def validate_change_pct(change_pct: float, ticker: str = "") -> bool:
    """Validate a daily change percentage. EGX has a ±20% daily limit."""
    if change_pct is None:
        return False
    try:
        c = float(change_pct)
        return bool(-20.0 <= c <= 20.0 and np.isfinite(c))
    except (TypeError, ValueError):
        return False


def deduplicate_stocks(stocks: list[dict]) -> list[dict]:
    """Remove duplicate stocks by symbol, keeping the first occurrence."""
    seen = set()
    result = []
    for s in stocks:
        sym = s.get("symbol", "")
        if sym and sym not in seen:
            seen.add(sym)
            result.append(s)
    return result
