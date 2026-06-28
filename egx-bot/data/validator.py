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
from datetime import datetime, timedelta, timezone
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
            self.quality_score = max(0.0, self.quality_score - 0.15)


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

        try:
            inf_count = int(np.isinf(df[col].astype(float)).sum())
        except (TypeError, ValueError):
            inf_count = 0
        if inf_count > 0:
            result.add_issue(f"{ticker}: {inf_count} inf values in {col}", "error")

    # 4. Check for negative or zero prices
    for col in ["Open", "High", "Low", "Close"]:
        negative = (df[col].astype(float) <= 0).sum()
        if negative > 0:
            result.add_issue(f"{ticker}: {negative} non-positive values in {col}", "error")

    # 5. Check price consistency: High >= Low, High >= Open, High >= Close, etc.
    # Vectorized for performance (O(1) instead of O(n) iterrows)
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    inconsistent = int(((h < l) | (h < o) | (h < c) | (l > o) | (l > c)).sum())

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

    now = datetime.now(timezone.utc)
    # Handle both tz-aware and naive datetimes
    if last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
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


# ─── Stage 1: Enhanced Data Validation ───────────────────────────────────────
# These functions add rigorous post-scrape validation and error classification.

@dataclass
class ScrapedPriceResult:
    """Result of validating a scraped stock price."""
    is_valid: bool
    is_suspicious: bool
    reason: str = ""

    @property
    def status(self) -> str:
        if not self.is_valid:
            return "rejected"
        if self.is_suspicious:
            return "suspicious"
        return "valid"


def validate_scraped_price(
    price: float | None,
    last_known_price: float | None = None,
    ticker: str = "",
) -> ScrapedPriceResult:
    """
    Validate a scraped stock price with deviation check.
    
    Checks:
    - Price is not None, not zero, not NaN/inf
    - If last_known_price is provided, checks if deviation > 4%
      (accounts for natural timing differences between scrapers)
    - Stocks exceeding 4% deviation: is_valid=True, is_suspicious=True
      → placed in "Needs Verification" list, excluded from final scoring
    
    Returns ScrapedPriceResult with is_valid, is_suspicious, and reason.
    """
    # 1. Null/None check
    if price is None:
        return ScrapedPriceResult(
            is_valid=False, is_suspicious=False,
            reason=f"{ticker}: price is null/None — rejected",
        )

    # 2. Numeric check
    try:
        p = float(price)
    except (TypeError, ValueError):
        return ScrapedPriceResult(
            is_valid=False, is_suspicious=False,
            reason=f"{ticker}: price '{price}' is not numeric — rejected",
        )

    # 3. Zero or negative
    if p <= 0:
        return ScrapedPriceResult(
            is_valid=False, is_suspicious=False,
            reason=f"{ticker}: price {p} is zero or negative — rejected",
        )

    # 4. NaN/inf
    if not np.isfinite(p):
        return ScrapedPriceResult(
            is_valid=False, is_suspicious=False,
            reason=f"{ticker}: price {p} is not finite (NaN/inf) — rejected",
        )

    # 5. Deviation check — compare with last known close
    # Threshold: 4% — accounts for natural timing differences between scrapers
    # (stockanalysis.com EOD vs tvDatafeed last close).
    # Stocks exceeding 4% go to "Needs Verification" — not rejected, but excluded from scoring.
    PRICE_DEVIATION_THRESHOLD = 4.0  # percent

    if last_known_price is not None and last_known_price > 0:
        try:
            lkp = float(last_known_price)
            if lkp > 0 and np.isfinite(lkp):
                deviation_pct = abs(p - lkp) / lkp * 100
                if deviation_pct > PRICE_DEVIATION_THRESHOLD:
                    return ScrapedPriceResult(
                        is_valid=True, is_suspicious=True,
                        reason=(
                            f"{ticker}: price {p} deviates {deviation_pct:.1f}% "
                            f"from last close {lkp} — Needs Verification (exceeds {PRICE_DEVIATION_THRESHOLD}% threshold), excluded from scoring"
                        ),
                    )
        except (TypeError, ValueError):
            pass  # Can't compare, accept the price

    return ScrapedPriceResult(is_valid=True, is_suspicious=False, reason="")


def classify_scrape_error(error: Exception, source: str = "") -> str:
    """
    Classify a scraping/downloading error into a specific type.
    Returns one of: 'timeout', 'no_response', 'page_structure_changed',
    'auth_failure', 'rate_limited', 'insufficient_data', 'unknown'.
    """
    err_str = str(error).lower()

    if "timeout" in err_str or "timed out" in err_str:
        return "timeout"
    if "connection" in err_str or "refused" in err_str or "unreachable" in err_str:
        return "no_response"
    if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "forbidden" in err_str:
        return "auth_failure"
    if "429" in err_str or "rate" in err_str or "too many" in err_str:
        return "rate_limited"
    if "no table" in err_str or "no tbody" in err_str or "not found" in err_str:
        return "page_structure_changed"
    if "insufficient" in err_str or "not enough" in err_str:
        return "insufficient_data"

    return "unknown"


def check_scraper_freshness(date_str: str | None) -> tuple[bool, str]:
    """
    Check if scraper data is fresh enough based on a date string.
    Returns (is_fresh, reason).
    """
    if not date_str:
        return False, "no date available from scraper"

    try:
        # Try common date formats
        from datetime import datetime as dt
        for fmt in ("%Y-%m-%d", "%b %d %Y", "%B %d %Y", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                scraped_date = dt.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue
        else:
            return False, f"could not parse date '{date_str}'"

        now = dt.now(timezone.utc)
        if scraped_date.tzinfo is None:
            scraped_date = scraped_date.replace(tzinfo=timezone.utc)

        age_hours = (now - scraped_date).total_seconds() / 3600
        if age_hours < 0:
            return True, "data is from today or future"
        if age_hours < FRESH_THRESHOLD_HOURS:
            return True, f"data is {age_hours:.0f}h old (fresh)"
        if age_hours < STALE_THRESHOLD_HOURS:
            return False, f"data is {age_hours:.0f}h old (stale, exceeds {FRESH_THRESHOLD_HOURS}h threshold)"
        return False, f"data is {age_hours:.0f}h old (very stale)"

    except Exception as e:
        return False, f"freshness check failed: {e}"


@dataclass
class ScanStatus:
    """Tracks the status of the last scan operation for reporting."""
    source: str = "unknown"
    failed_sources: list[str] = field(default_factory=list)
    error_type: str = ""  # classified error type
    error_detail: str = ""
    used_fallback: bool = False
    fallback_date: str = ""
    total_scraped: int = 0
    total_validated: int = 0
    total_rejected: int = 0
    suspicious_count: int = 0
    suspicious_tickers: list[str] = field(default_factory=list)
    needs_verification: list[str] = field(default_factory=list)
    no_indicators_tickers: list[str] = field(default_factory=list)
    limited_coverage: bool = False
    coverage_count: int = 0
    has_reliable_data: bool = False
    last_successful_date: str = ""

    def summary(self) -> str:
        """Human-readable summary for logging."""
        parts = [f"source={self.source}", f"scraped={self.total_scraped}", f"validated={self.total_validated}"]
        if self.total_rejected:
            parts.append(f"rejected={self.total_rejected}")
        if self.suspicious_count:
            parts.append(f"suspicious={self.suspicious_count} ({', '.join(self.suspicious_tickers[:5])})")
        if self.needs_verification:
            parts.append(f"needs_verification={len(self.needs_verification)} ({', '.join(self.needs_verification[:5])})")
        if self.no_indicators_tickers:
            parts.append(f"no_indicators={len(self.no_indicators_tickers)}")
        if self.failed_sources:
            parts.append(f"failed={','.join(self.failed_sources)}")
        if self.used_fallback:
            parts.append(f"fallback_date={self.fallback_date}")
        if self.limited_coverage:
            parts.append(f"limited_coverage={self.coverage_count}")
        return " | ".join(parts)
