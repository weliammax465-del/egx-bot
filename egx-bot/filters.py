"""
filters.py
----------
Pre-scoring exclusion filters and scoring factors for the EGX Bot
Liquidity-First Strategy v2.

Pipeline order:
  1. Liquidity Gate (exclude if turnover < 1M EGP)
  2. Price Limit Gate (exclude if |change| >= 10%)
  ─── indicators computed after gates ───
  3. Volume Surge (scoring factor — today's vol > each of last 3 days)
  4. Two-day Confirmation (scoring factor — surge for 2 days OR pullback after strong day)
  5. Trend & Relative Strength Filter (exclude if price < EMA50, RSI > 70, or RS < EGX30)
  6. Risk Management Filter (exclude if R/R < 2:1 using ATR-based stop-loss)
  ─── final scoring ───
  7. Score: Liquidity 30% + Trend/RS 25% + Confirmation 20% + R/R 25%

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


# ─── Exclusion Reason Codes (for RecommendationHistory analytics) ────────────

EXCLUSION_LOW_LIQUIDITY = "سيولة ضعيفة"
EXCLUSION_NO_LIQUIDITY_DATA = "بيانات سيولة غير متوفرة"
EXCLUSION_PRICE_LIMIT = "وصل لحد التذبذب اليومي"
EXCLUSION_INSUFFICIENT_DATA = "بيانات غير كافية"
EXCLUSION_NO_CONFIRMATION = "لا يوجد تأكيد يومين"
EXCLUSION_WEAK_TREND = "اتجاه نازل أو قوة نسبية ضعيفة"
EXCLUSION_POOR_RISK_REWARD = "risk-reward غير كافٍ"


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of a pre-scoring gate check."""
    passed: bool
    gate_name: str
    reason: str              # Arabic explanation
    reason_en: str           # English explanation (for logs)
    exclusion_code: str = "" # short code for RecommendationHistory analytics
    details: dict = field(default_factory=dict)


@dataclass
class SurgeResult:
    """Result of the Volume Surge check (step 3)."""
    surged: bool                    # True if today's vol > each of last 3 days
    days_surged: int                # How many of the last 3 days were surpassed
    today_volume: float
    comparison_volumes: list[float] # the 3 prior day volumes
    reason_ar: str
    reason_en: str


@dataclass
class ConfirmationResult:
    """Result of the Two-day Confirmation check (step 4)."""
    confirmed: bool
    confirmation_type: str    # "two_day_surge" | "pullback_after_strong" | "none"
    reason_ar: str
    reason_en: str
    exclusion_code: str = ""
    details: dict = field(default_factory=dict)


# ─── Step 1: Liquidity Gate ──────────────────────────────────────────────────

def _has_valid_volume(df: pd.DataFrame, avg_days: int) -> tuple[bool, int]:
    """Check if Volume column has usable data (>= 50% non-zero, non-NaN)."""
    vol_tail = df["Volume"].tail(avg_days)
    valid_mask = (vol_tail > 0) & vol_tail.notna() & np.isfinite(vol_tail)
    valid_count = int(valid_mask.sum())
    has_valid = valid_count >= (avg_days // 2)
    return has_valid, valid_count


def liquidity_gate(
    df: pd.DataFrame,
    ticker: str = "",
    avg_days: int = None,
) -> GateResult:
    """
    Check if a stock meets the minimum liquidity requirement.
    Turnover = Volume × Close price (in EGP), averaged over `avg_days`.

    Three outcomes:
      1. PASS — valid volume, turnover >= MIN_TURNOVER_EGP
      2. FAIL "low liquidity" — valid volume, turnover < MIN_TURNOVER_EGP
      3. FAIL "liquidity data unavailable" — Volume is zeros/NaN
    """
    if avg_days is None:
        avg_days = config.MIN_TURNOVER_AVG_DAYS

    if df is None or len(df) < avg_days:
        return GateResult(
            passed=False,
            gate_name="liquidity",
            reason=f"{EXCLUSION_INSUFFICIENT_DATA} لحساب متوسط السيولة (متاح {len(df) if df is not None else 0} يوم، مطلوب {avg_days})",
            reason_en=f"insufficient data for liquidity avg ({len(df) if df is not None else 0} < {avg_days} days)",
            exclusion_code=EXCLUSION_INSUFFICIENT_DATA,
            details={"available_days": len(df) if df is not None else 0, "required_days": avg_days},
        )

    has_valid_vol, valid_vol_days = _has_valid_volume(df, avg_days)

    if not has_valid_vol:
        return GateResult(
            passed=False,
            gate_name="liquidity",
            reason=f"{EXCLUSION_NO_LIQUIDITY_DATA} — عمود الحجم يحتوي {valid_vol_days}/{avg_days} قيم صالحة فقط",
            reason_en=f"liquidity data unavailable — Volume column has only {valid_vol_days}/{avg_days} valid values",
            exclusion_code=EXCLUSION_NO_LIQUIDITY_DATA,
            details={"valid_volume_days": valid_vol_days, "required_days": avg_days, "min_valid_ratio": 0.5},
        )

    vol_tail = df["Volume"].tail(avg_days)
    close_tail = df["Close"].tail(avg_days)
    turnover_series = (vol_tail * close_tail).replace([np.inf, -np.inf], np.nan).dropna()
    avg_turnover = float(turnover_series.mean())
    latest_turnover = float((df["Volume"].iloc[-1] * df["Close"].iloc[-1]))

    passed = avg_turnover >= config.MIN_TURNOVER_EGP

    if passed:
        return GateResult(
            passed=True, gate_name="liquidity",
            reason=f"سيولة كافية (متوسط {avg_turnover/1e6:.2f}M EGP/يوم)",
            reason_en=f"liquidity OK (avg {avg_turnover/1e6:.2f}M EGP/day)",
            details={"avg_turnover_egp": round(avg_turnover, 2), "latest_turnover_egp": round(latest_turnover, 2),
                     "min_required_egp": config.MIN_TURNOVER_EGP, "avg_days": avg_days, "valid_volume_days": valid_vol_days},
        )
    else:
        return GateResult(
            passed=False, gate_name="liquidity",
            reason=f"{EXCLUSION_LOW_LIQUIDITY} (متوسط {avg_turnover/1e6:.2f}M EGP/يوم، الحد الأدنى {config.MIN_TURNOVER_EGP/1e6:.1f}M)",
            reason_en=f"low liquidity (avg {avg_turnover/1e6:.2f}M EGP/day < {config.MIN_TURNOVER_EGP/1e6:.1f}M minimum)",
            exclusion_code=EXCLUSION_LOW_LIQUIDITY,
            details={"avg_turnover_egp": round(avg_turnover, 2), "latest_turnover_egp": round(latest_turnover, 2),
                     "min_required_egp": config.MIN_TURNOVER_EGP, "avg_days": avg_days, "valid_volume_days": valid_vol_days},
        )


# ─── Step 2: Price Limit Gate ────────────────────────────────────────────────

def price_limit_gate(daily_change_pct: float, ticker: str = "") -> GateResult:
    """Exclude stocks at EGX daily price limit (±10%)."""
    if daily_change_pct is None or not np.isfinite(daily_change_pct):
        return GateResult(
            passed=False, gate_name="price_limit",
            reason="نسبة التغير غير متاحة أو غير صالحة",
            reason_en="change_pct is None or non-finite",
            exclusion_code="نسبة التغير غير متاحة",
            details={"change_pct": daily_change_pct},
        )

    abs_change = abs(float(daily_change_pct))
    passed = abs_change < config.PRICE_LIMIT_THRESHOLD_PCT

    if passed:
        return GateResult(
            passed=True, gate_name="price_limit",
            reason=f"التغير {daily_change_pct:+.1f}% (ضمن الحد الطبيعي)",
            reason_en=f"change {daily_change_pct:+.1f}% (within normal range)",
            details={"change_pct": daily_change_pct, "threshold": config.PRICE_LIMIT_THRESHOLD_PCT},
        )
    else:
        direction = "صعودي" if daily_change_pct > 0 else "هبوطي"
        return GateResult(
            passed=False, gate_name="price_limit",
            reason=f"{EXCLUSION_PRICE_LIMIT} {direction} ({daily_change_pct:+.1f}%) — السهم مجمد، لا يمكن التنفيذ",
            reason_en=f"price limit hit ({daily_change_pct:+.1f}%) — stock frozen, not tradeable",
            exclusion_code=EXCLUSION_PRICE_LIMIT,
            details={"change_pct": daily_change_pct, "threshold": config.PRICE_LIMIT_THRESHOLD_PCT},
        )


# ─── Steps 1+2: Combined Pre-scoring Gate ────────────────────────────────────

def pass_all_gates(
    df: pd.DataFrame, daily_change_pct: float, ticker: str = "",
) -> tuple[bool, list[GateResult]]:
    """Run liquidity + price limit gates. Fail fast on first failure."""
    results: list[GateResult] = []

    liq = liquidity_gate(df, ticker)
    results.append(liq)
    if not liq.passed:
        logger.info(f"  🚫 {ticker}: EXCLUDED — {liq.reason_en}")
        return False, results

    pl = price_limit_gate(daily_change_pct, ticker)
    results.append(pl)
    if not pl.passed:
        logger.info(f"  🚫 {ticker}: EXCLUDED — {pl.reason_en}")
        return False, results

    logger.debug(f"  ✅ {ticker}: passed all gates (liquidity + price limit)")
    return True, results


# ─── Step 3: Volume Surge ────────────────────────────────────────────────────

def volume_surge_check(df: pd.DataFrame, ticker: str = "") -> SurgeResult:
    """
    Check if today's volume is higher than EACH of the last 3 trading days
    individually (not the average).

    This is a scoring factor, not a gate — it boosts the score but doesn't
    exclude on its own. The exclusion happens in the confirmation check (step 4)
    if the surge doesn't persist for 2 days.

    Args:
        df: OHLCV DataFrame with 'Volume' column.
        ticker: Stock symbol (for logging).

    Returns:
        SurgeResult with surged=True if today's vol > all 3 prior days.
    """
    lookback = config.VOLUME_SURGE_LOOKBACK_DAYS

    if df is None or len(df) < lookback + 1:
        return SurgeResult(
            surged=False, days_surged=0,
            today_volume=0, comparison_volumes=[],
            reason_ar="بيانات غير كافية لفحص Volume Surge",
            reason_en="insufficient data for volume surge check",
        )

    today_vol = float(df["Volume"].iloc[-1])
    # Get the 3 days BEFORE today (not including today)
    prior_vols = [float(df["Volume"].iloc[-(i + 2)]) for i in range(lookback)]

    # Count how many of the prior 3 days today's volume surpassed
    days_surged = sum(1 for v in prior_vols if today_vol > v)
    surged = days_surged == lookback  # must beat ALL 3 days

    if surged:
        reason_ar = f"حجم اليوم ({today_vol:,.0f}) أعلى من آخر {lookback} جلسات منفصلة"
        reason_en = f"volume surge: today ({today_vol:,.0f}) > all {lookback} prior days"
    else:
        reason_ar = f"حجم اليوم ({today_vol:,.0f}) تفوق على {days_surged}/{lookback} جلسات سابقة فقط"
        reason_en = f"no surge: today ({today_vol:,.0f}) > {days_surged}/{lookback} prior days"

    return SurgeResult(
        surged=surged, days_surged=days_surged,
        today_volume=today_vol, comparison_volumes=prior_vols,
        reason_ar=reason_ar, reason_en=reason_en,
    )


# ─── Step 4: Two-day Confirmation ────────────────────────────────────────────

def confirmation_check(df: pd.DataFrame, ticker: str = "") -> ConfirmationResult:
    """
    Require the stock to show sustained strength, not just a one-day spike.

    Passes if EITHER:
      A) Volume surge for at least 2 consecutive days (yesterday AND today both
         had volume > their preceding 3 days), OR
      B) Today is a light pullback (small decline < 2%) after a strong up day
         yesterday (gain >= 3%) — this indicates profit-taking after a real
         move, not FOMO entry at the top.

    Fails if the stock is surging for the first time today only with no
    prior confirmation — this prevents buying at the peak of a spike.

    Args:
        df: OHLCV DataFrame with 'Volume' and 'Close' columns.
        ticker: Stock symbol (for logging).

    Returns:
        ConfirmationResult with confirmed=True if either condition is met.
    """
    if df is None or len(df) < 5:
        return ConfirmationResult(
            confirmed=False, confirmation_type="none",
            reason_ar="بيانات غير كافية للتأكيد عبر يومين",
            reason_en="insufficient data for 2-day confirmation",
        )

    # ── Condition A: Two-day volume surge ──
    # Check if today AND yesterday both surged vs their prior 3 days
    today_surge = volume_surge_check(df, ticker)
    # For yesterday: use df without the last row
    yesterday_df = df.iloc[:-1]
    yesterday_surge = volume_surge_check(yesterday_df, ticker)

    if today_surge.surged and yesterday_surge.surged:
        return ConfirmationResult(
            confirmed=True, confirmation_type="two_day_surge",
            reason_ar="تأكيد سيولة ليومين متتاليين (Volume Surge مستمر)",
            reason_en="confirmed: 2-day consecutive volume surge",
            details={"today_vol": today_surge.today_volume, "yesterday_vol": yesterday_surge.today_volume},
        )

    # ── Condition B: Pullback after strong up day ──
    # Yesterday must have gained >= 3%, today must be a small decline (< 2%)
    if len(df) >= 3:
        today_close = float(df["Close"].iloc[-1])
        yesterday_close = float(df["Close"].iloc[-2])
        day_before_close = float(df["Close"].iloc[-3])

        yesterday_change = ((yesterday_close - day_before_close) / day_before_close) * 100
        today_change = ((today_close - yesterday_close) / yesterday_close) * 100

        # Strong up day yesterday (>= 3%), light pullback today (between -2% and 0%)
        is_strong_yesterday = yesterday_change >= 3.0
        is_light_pullback = -2.0 <= today_change < 0.0

        if is_strong_yesterday and is_light_pullback:
            return ConfirmationResult(
                confirmed=True, confirmation_type="pullback_after_strong",
                reason_ar=f"تراجع طفيف ({today_change:+.1f}%) بعد يوم ارتفاع قوي ({yesterday_change:+.1f}%)",
                reason_en=f"pullback ({today_change:+.1f}%) after strong day ({yesterday_change:+.1f}%)",
                details={"yesterday_change": round(yesterday_change, 2), "today_change": round(today_change, 2)},
            )

    # ── Neither condition met ──
    return ConfirmationResult(
        confirmed=False, confirmation_type="none",
        reason_ar=f"{EXCLUSION_NO_CONFIRMATION} — الحركة أول مرة اليوم بدون تأكيد سابق",
        reason_en=f"no 2-day confirmation — first-day spike without prior support",
        exclusion_code=EXCLUSION_NO_CONFIRMATION,
        details={"today_surge": today_surge.surged, "yesterday_surge": yesterday_surge.surged},
    )


# ─── Step 5: Trend & Relative Strength Filter ────────────────────────────────

def trend_strength_filter(
    df: pd.DataFrame,
    ema50: float,
    rsi: float,
    egx30_change_20d: float,
    ticker: str = "",
) -> GateResult:
    """
    Post-indicator filter: exclude stocks with weak trend or no relative strength.

    Three conditions (ALL must pass):
      1. Price above EMA50 (medium-term uptrend)
      2. RSI < 70 (not overbought — avoid entering after a sharp rally)
      3. 20-day stock change > 20-day EGX 30 change (relative strength)

    Args:
        df: OHLCV DataFrame (for computing 20-day change if needed).
        ema50: The 50-day EMA value from indicators.
        rsi: The current RSI value from indicators.
        egx30_change_20d: EGX 30 index % change over 20 trading days.
        ticker: Stock symbol (for logging).

    Returns:
        GateResult with passed=True if all three conditions are met.
    """
    if df is None or len(df) < 20:
        return GateResult(
            passed=False, gate_name="trend_strength",
            reason=f"{EXCLUSION_INSUFFICIENT_DATA} لفلتر الاتجاه والقوة النسبية",
            reason_en="insufficient data for trend/RS filter",
            exclusion_code=EXCLUSION_INSUFFICIENT_DATA,
        )

    current_price = float(df["Close"].iloc[-1])
    price_20d_ago = float(df["Close"].iloc[-20])
    stock_change_20d = ((current_price - price_20d_ago) / price_20d_ago) * 100

    failures = []

    # Condition 1: Price > EMA50
    if ema50 is None or ema50 <= 0:
        failures.append("EMA50 غير متاح")
    elif current_price < ema50:
        failures.append(f"السعر ({current_price:.2f}) تحت EMA50 ({ema50:.2f}) — اتجاه هابط متوسط المدى")

    # Condition 2: RSI < 70 (not overbought)
    if rsi is None or not np.isfinite(rsi):
        failures.append("RSI غير متاح")
    elif rsi >= config.RSI_OVERBOUGHT:
        failures.append(f"RSI={rsi:.0f} (> {config.RSI_OVERBOUGHT:.0f}) — تشبع شرائي، الدخول متأخر")

    # Condition 3: Relative strength vs EGX 30
    if egx30_change_20d is None:
        failures.append("تغير مؤشر EGX 30 غير متاح للمقارنة")
    elif stock_change_20d <= egx30_change_20d:
        failures.append(
            f"السهم ({stock_change_20d:+.1f}% في 20 يوم) أضعف من EGX 30 ({egx30_change_20d:+.1f}%) — قوة نسبية ضعيفة"
        )

    if failures:
        return GateResult(
            passed=False, gate_name="trend_strength",
            reason=f"{EXCLUSION_WEAK_TREND}: {'، '.join(failures)}",
            reason_en=f"trend/RS failed: {'; '.join(failures)}",
            exclusion_code=EXCLUSION_WEAK_TREND,
            details={
                "price": round(current_price, 2), "ema50": round(ema50, 2) if ema50 else 0,
                "rsi": round(rsi, 1) if rsi else 0, "stock_change_20d": round(stock_change_20d, 2),
                "egx30_change_20d": round(egx30_change_20d, 2) if egx30_change_20d else 0,
            },
        )

    return GateResult(
        passed=True, gate_name="trend_strength",
        reason=f"اتجاه صاعد + RSI={rsi:.0f} + قوة نسبية ({stock_change_20d:+.1f}% vs EGX30 {egx30_change_20d:+.1f}%)",
        reason_en=f"trend OK: price>EMA50, RSI={rsi:.0f}, RS={stock_change_20d:+.1f}%>EGX30 {egx30_change_20d:+.1f}%",
        details={
            "price": round(current_price, 2), "ema50": round(ema50, 2),
            "rsi": round(rsi, 1), "stock_change_20d": round(stock_change_20d, 2),
            "egx30_change_20d": round(egx30_change_20d, 2),
        },
    )


# ─── Step 6: Risk Management Filter ──────────────────────────────────────────

def risk_filter(
    current_price: float,
    atr: float,
    resistance: float,
    ticker: str = "",
) -> GateResult:
    """
    Exclude stocks with insufficient risk/reward ratio.

    Stop-loss = current_price - (ATR_STOP_LOSS_MULTIPLIER × ATR)
    Target = resistance level (nearest overhead resistance)
    R/R = (resistance - price) / (price - stop_loss)

    If R/R < MIN_RISK_REWARD_RATIO (2.0), the stock is excluded — even if it
    passed all other filters. No trade is better than a bad-risk trade.

    Args:
        current_price: Live/current stock price.
        atr: Current ATR value from indicators.
        resistance: Nearest resistance level above price.
        ticker: Stock symbol (for logging).

    Returns:
        GateResult with passed=True if R/R >= 2.0.
    """
    # Need valid inputs
    if current_price is None or current_price <= 0:
        return GateResult(
            passed=False, gate_name="risk",
            reason="السعر غير متاح لحساب المخاطرة",
            reason_en="price unavailable for risk calculation",
            exclusion_code=EXCLUSION_POOR_RISK_REWARD,
        )

    if atr is None or atr <= 0 or not np.isfinite(atr):
        return GateResult(
            passed=False, gate_name="risk",
            reason="ATR غير متاح لحساب وقف الخسارة",
            reason_en="ATR unavailable for stop-loss calculation",
            exclusion_code=EXCLUSION_POOR_RISK_REWARD,
        )

    # Compute stop-loss
    stop_loss = current_price - (config.ATR_STOP_LOSS_MULTIPLIER * atr)
    risk_per_share = current_price - stop_loss  # = 1.5 × ATR

    if risk_per_share <= 0:
        return GateResult(
            passed=False, gate_name="risk",
            reason="وقف الخسارة فوق السعر الحالي — خطأ في البيانات",
            reason_en="stop-loss above current price — data error",
            exclusion_code=EXCLUSION_POOR_RISK_REWARD,
            details={"price": current_price, "atr": atr, "stop_loss": stop_loss},
        )

    # Compute target and R/R from resistance
    if resistance is None or resistance <= current_price:
        # No resistance above price → can't compute R/R
        return GateResult(
            passed=False, gate_name="risk",
            reason=f"{EXCLUSION_POOR_RISK_REWARD} — لا توجد مقاومة فوق السعر لتحديد الهدف",
            reason_en=f"poor R/R — no resistance above price for target",
            exclusion_code=EXCLUSION_POOR_RISK_REWARD,
            details={"price": current_price, "atr": atr, "stop_loss": round(stop_loss, 2), "resistance": resistance or 0},
        )

    reward_per_share = resistance - current_price
    rr_ratio = reward_per_share / risk_per_share

    if rr_ratio >= config.MIN_RISK_REWARD_RATIO:
        return GateResult(
            passed=True, gate_name="risk",
            reason=f"نسبة مخاطرة/عائد {rr_ratio:.1f}:1 (وقف {stop_loss:.2f}، هدف {resistance:.2f})",
            reason_en=f"R/R OK: {rr_ratio:.1f}:1 (stop={stop_loss:.2f}, target={resistance:.2f})",
            details={
                "price": round(current_price, 2), "atr": round(atr, 4),
                "stop_loss": round(stop_loss, 2), "target": round(resistance, 2),
                "risk_per_share": round(risk_per_share, 2), "reward_per_share": round(reward_per_share, 2),
                "rr_ratio": round(rr_ratio, 2),
            },
        )
    else:
        return GateResult(
            passed=False, gate_name="risk",
            reason=f"{EXCLUSION_POOR_RISK_REWARD} ({rr_ratio:.1f}:1، الحد الأدنى {config.MIN_RISK_REWARD_RATIO:.1f}:1)",
            reason_en=f"poor R/R: {rr_ratio:.1f}:1 < {config.MIN_RISK_REWARD_RATIO:.1f}:1 minimum",
            exclusion_code=EXCLUSION_POOR_RISK_REWARD,
            details={
                "price": round(current_price, 2), "atr": round(atr, 4),
                "stop_loss": round(stop_loss, 2), "target": round(resistance, 2),
                "risk_per_share": round(risk_per_share, 2), "reward_per_share": round(reward_per_share, 2),
                "rr_ratio": round(rr_ratio, 2),
            },
        )


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_exclusion_reason(results: list[GateResult]) -> str:
    """Extract the Arabic reason from the first failing gate."""
    for r in results:
        if not r.passed:
            return r.reason
    return ""


def get_exclusion_code(results: list[GateResult]) -> str:
    """Extract the short exclusion code for RecommendationHistory analytics."""
    for r in results:
        if not r.passed:
            return r.exclusion_code
    return ""
