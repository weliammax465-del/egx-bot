"""
scoring.py — Liquidity-First Strategy v2
-----------------------------------------
Deterministic scoring engine (0-100) for EGX stock analysis.

NEW SCORING ARCHITECTURE (v2):
  Scoring only happens for stocks that passed ALL pre-scoring filters
  (liquidity gate, price limit, trend/RS, risk management).

  Final score weights:
    Liquidity Strength     30%
    Trend & Relative Str.  25%
    Two-day Confirmation   20%
    Risk/Reward            25%

  Other indicators (MACD, Bollinger, ADX, OBV, Stochastic RSI) are computed
  and displayed as "additional context" in stock details — they do NOT
  enter the final score or the accept/reject decision.

No AI is involved in scoring. AI only explains the computed results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

import config
from indicators import StockAnalysis, IndicatorResult
from filters import (
    SurgeResult, ConfirmationResult, GateResult,
    volume_surge_check, confirmation_check,
)

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ScoreFactor:
    """A single factor contributing to the total score."""
    name: str
    name_ar: str
    raw_score: float       # 0-100 for this factor
    weight: float          # relative weight (sums to 100)
    weighted_score: float  # raw_score * weight / 100
    reason: str            # explanation in Arabic


@dataclass
class ScoringResult:
    """Complete scoring result for a stock."""
    total_score: int
    recommendation: str           # "Buy", "Watch", "Sell", "No Trade"
    recommendation_ar: str
    factors: list[ScoreFactor] = field(default_factory=list)
    risk_level: str = "Medium"
    risk_reason: str = ""
    data_quality: float = 1.0
    data_freshness: str = "fresh"
    pass_reasons: list[str] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)
    # v2 additions
    exclusion_reason: str = ""    # Arabic reason if excluded (empty if scored)
    exclusion_code: str = ""      # short code for analytics
    stop_loss: float = 0.0
    target: float = 0.0
    rr_ratio: float = 0.0
    surge_info: str = ""          # volume surge description
    confirmation_info: str = ""   # confirmation type description

    @property
    def is_actionable(self) -> bool:
        return self.recommendation in ("Buy", "Sell")


# ─── Factor Weights (must sum to 100) — Liquidity-First v2 ───────────────────

WEIGHTS = {
    "liquidity": 30,       # Liquidity strength (turnover magnitude)
    "trend_rs": 25,        # Trend quality + relative strength vs EGX 30
    "confirmation": 20,    # Two-day volume surge confirmation
    "risk_reward": 25,     # Risk/Reward ratio quality
}

assert sum(WEIGHTS.values()) == 100, "Factor weights must sum to 100"


# ─── Helper ──────────────────────────────────────────────────────────────────

def _get_indicator(analysis: StockAnalysis, name: str) -> Optional[IndicatorResult]:
    """Find an indicator by name."""
    for ind in analysis.indicators:
        if ind.name == name:
            return ind
    return None


# ─── Scoring Functions (v2) ──────────────────────────────────────────────────

def _score_liquidity_strength(
    avg_turnover_egp: float,
    latest_turnover_egp: float,
) -> ScoreFactor:
    """
    Liquidity Strength (30%): how strong is the stock's liquidity?

    Score is based on how far above the minimum threshold the stock's
    average turnover is. Stocks barely above 1M get a low score;
    stocks with 50M+ turnover get near-maximum.

    Scale:
      1M (minimum)     → 50 (barely passes)
      5M               → 65
      10M              → 72
      50M              → 87
      100M+            → 95+

    Also considers latest day vs average (rising liquidity = bonus).
    """
    min_t = config.MIN_TURNOVER_EGP

    # Logarithmic scale: rewards higher turnover but with diminishing returns
    # log(1M) → 0, log(100M) → ~4.6, log(1B) → ~6.9
    import math
    if avg_turnover_egp <= 0:
        base_score = 50.0
    else:
        ratio = avg_turnover_egp / min_t  # 1.0 = minimum, 100.0 = 100× minimum
        # log scale: ratio=1 → 50, ratio=10 → 70, ratio=100 → 87, ratio=500 → 95
        base_score = 50 + min(45, math.log10(max(ratio, 1.0)) * 18.5)

    # Bonus: if latest turnover > 20-day average (rising liquidity)
    if avg_turnover_egp > 0 and latest_turnover_egp > avg_turnover_egp * 1.3:
        base_score = min(100, base_score + 5)
        reason_suffix = " + سيولة اليوم أعلى من المتوسط"
    elif avg_turnover_egp > 0 and latest_turnover_egp < avg_turnover_egp * 0.7:
        base_score = max(0, base_score - 5)
        reason_suffix = " — سيولة اليوم أقل من المتوسط"
    else:
        reason_suffix = ""

    score = max(0, min(100, base_score))

    return ScoreFactor(
        name="Liquidity Strength", name_ar="قوة السيولة",
        raw_score=score, weight=WEIGHTS["liquidity"],
        weighted_score=score * WEIGHTS["liquidity"] / 100,
        reason=f"متوسط السيولة {avg_turnover_egp/1e6:.2f}M EGP/يوم ({avg_turnover_egp/min_t:.0f}× الحد الأدنى){reason_suffix}",
    )


def _score_trend_rs(
    price: float,
    ema50: float,
    rsi: float,
    stock_change_20d: float,
    egx30_change_20d: float,
) -> ScoreFactor:
    """
    Trend & Relative Strength (25%): quality of the uptrend + RS vs EGX 30.

    Components:
      - Price above EMA50 (uptrend confirmation) — up to 40 points
      - RSI in healthy zone (40-65, not overbought) — up to 25 points
      - Relative strength vs EGX 30 — up to 35 points
    """
    score = 0.0
    reasons = []

    # ── Trend: Price vs EMA50 (40 pts max) ──
    if ema50 and ema50 > 0 and price > 0:
        pct_above = ((price - ema50) / ema50) * 100
        if pct_above > 10:
            score += 40
            reasons.append(f"السعر فوق EMA50 بـ {pct_above:.1f}%")
        elif pct_above > 5:
            score += 35
            reasons.append(f"السعر فوق EMA50 بـ {pct_above:.1f}%")
        elif pct_above > 0:
            score += 28
            reasons.append(f"السعر فوق EMA50 بـ {pct_above:.1f}%")
        else:
            score += 10
            reasons.append(f"السعر تحت EMA50 بـ {abs(pct_above):.1f}%")
    else:
        score += 15
        reasons.append("EMA50 غير متاح")

    # ── RSI health (25 pts max) ──
    if rsi is not None and np.isfinite(rsi):
        if 45 <= rsi <= 65:
            score += 25  # ideal zone — momentum without overbought
            reasons.append(f"RSI={rsi:.0f} (منطقة صحية)")
        elif 40 <= rsi < 45:
            score += 20
            reasons.append(f"RSI={rsi:.0f} (زخم معتدل)")
        elif 65 < rsi < 70:
            score += 15  # approaching overbought
            reasons.append(f"RSI={rsi:.0f} (قرب التشبع)")
        elif rsi >= 70:
            score += 5  # overbought (but passed the filter, so just barely)
            reasons.append(f"RSI={rsi:.0f} (تشبع شرائي)")
        else:
            score += 10
            reasons.append(f"RSI={rsi:.0f} (زخم ضعيف)")
    else:
        score += 12
        reasons.append("RSI غير متاح")

    # ── Relative Strength vs EGX 30 (35 pts max) ──
    if egx30_change_20d is not None:
        rs_diff = stock_change_20d - egx30_change_20d
        if rs_diff > 10:
            score += 35
            reasons.append(f"قوة نسبية ممتازة (+{rs_diff:.1f}% فوق EGX 30)")
        elif rs_diff > 5:
            score += 28
            reasons.append(f"قوة نسبية جيدة (+{rs_diff:.1f}% فوق EGX 30)")
        elif rs_diff > 0:
            score += 20
            reasons.append(f"قوة نسبية موجبة (+{rs_diff:.1f}% فوق EGX 30)")
        else:
            score += 5
            reasons.append(f"قوة نسبية سلبية ({rs_diff:.1f}% تحت EGX 30)")
    else:
        score += 15
        reasons.append("EGX 30 غير متاح للمقارنة")

    score = max(0, min(100, score))

    return ScoreFactor(
        name="Trend & RS", name_ar="الاتجاه والقوة النسبية",
        raw_score=score, weight=WEIGHTS["trend_rs"],
        weighted_score=score * WEIGHTS["trend_rs"] / 100,
        reason="، ".join(reasons),
    )


def _score_confirmation(confirmation: ConfirmationResult) -> ScoreFactor:
    """
    Two-day Confirmation (20%): has the stock shown sustained strength?

    Full score (90-100): 2-day consecutive volume surge
    Good score (70-80):  pullback after strong up day (healthy profit-taking)
    Low score (40-50):   surge today only, no yesterday confirmation
    No score (20-30):    no surge, no pullback pattern
    """
    if confirmation.confirmation_type == "two_day_surge":
        score = 95.0
        reason = "تأكيد سيولة ليومين متتاليين — قوة مستدامة"
    elif confirmation.confirmation_type == "pullback_after_strong":
        score = 75.0
        reason = confirmation.reason_ar
    elif confirmation.confirmed:
        score = 65.0
        reason = confirmation.reason_ar
    else:
        # Not confirmed — but stock still passed other filters
        score = 30.0
        reason = "لا يوجد تأكيد ليومين — حركة أول مرة"

    score = max(0, min(100, score))

    return ScoreFactor(
        name="Confirmation", name_ar="التأكيد عبر يومين",
        raw_score=score, weight=WEIGHTS["confirmation"],
        weighted_score=score * WEIGHTS["confirmation"] / 100,
        reason=reason,
    )


def _score_risk_reward(rr_ratio: float) -> ScoreFactor:
    """
    Risk/Reward (25%): quality of the trade's risk/reward profile.

    Scale:
      R/R >= 4:1 → 95
      R/R >= 3:1 → 85
      R/R >= 2:1 → 70 (minimum to pass the risk filter)
      R/R >= 1.5:1 → 55
      R/R < 1.5:1 → 30 (shouldn't happen — risk filter would exclude)
    """
    if rr_ratio >= 4.0:
        score = 95.0
        reason = f"نسبة ممتازة ({rr_ratio:.1f}:1)"
    elif rr_ratio >= 3.0:
        score = 85.0
        reason = f"نسبة قوية ({rr_ratio:.1f}:1)"
    elif rr_ratio >= 2.0:
        score = 70.0
        reason = f"نسبة جيدة ({rr_ratio:.1f}:1)"
    elif rr_ratio >= 1.5:
        score = 55.0
        reason = f"نسبة مقبولة ({rr_ratio:.1f}:1)"
    else:
        score = 30.0
        reason = f"نسبة ضعيفة ({rr_ratio:.1f}:1)"

    score = max(0, min(100, score))

    return ScoreFactor(
        name="Risk/Reward", name_ar="المخاطرة/العائد",
        raw_score=score, weight=WEIGHTS["risk_reward"],
        weighted_score=score * WEIGHTS["risk_reward"] / 100,
        reason=reason,
    )


# ─── Risk Assessment ─────────────────────────────────────────────────────────

def _assess_risk(analysis: StockAnalysis, rr_ratio: float) -> tuple[str, str]:
    """Determine risk level based on ATR and R/R ratio."""
    atr = _get_indicator(analysis, "ATR")
    reasons = []
    risk = "Medium"

    if atr:
        # ATR as percentage of price
        atr_pct = (atr.value / analysis.current_price) * 100 if analysis.current_price > 0 else 0
        if atr_pct > 5:
            risk = "High"
            reasons.append(f"تذبذب مرتفع ({atr_pct:.1f}%)")
        elif atr_pct < 2:
            risk = "Low"
            reasons.append(f"تذبذب منخفض ({atr_pct:.1f}%)")

    if rr_ratio < 2.5:
        reasons.append(f"نسبة مخاطرة/عائد متوسطة ({rr_ratio:.1f}:1)")
        if risk == "Low":
            risk = "Medium"
    elif rr_ratio >= 4:
        if risk == "Medium":
            risk = "Low"
        reasons.append(f"نسبة مخاطرة/عائد ممتازة ({rr_ratio:.1f}:1)")

    if not reasons:
        reasons.append("مخاطرة معتدلة")

    return risk, "، ".join(reasons)


# ─── Main Scoring Function (v2) ──────────────────────────────────────────────

def compute_score_v2(
    analysis: StockAnalysis,
    avg_turnover_egp: float,
    latest_turnover_egp: float,
    stock_change_20d: float,
    egx30_change_20d: float,
    confirmation: ConfirmationResult,
    rr_ratio: float,
    stop_loss: float,
    target: float,
    data_freshness: str = "fresh",
    data_quality: float = 1.0,
) -> ScoringResult:
    """
    Compute a deterministic 0-100 score using the Liquidity-First v2 weights.

    This function is called ONLY for stocks that passed ALL filters:
    liquidity gate, price limit, trend/RS, and risk management.

    The 4 scoring factors:
      1. Liquidity Strength    (30%) — turnover magnitude above minimum
      2. Trend & RS            (25%) — EMA50 + RSI zone + relative strength
      3. Two-day Confirmation  (20%) — sustained volume surge or pullback pattern
      4. Risk/Reward           (25%) — R/R ratio quality

    Args:
        analysis: StockAnalysis with computed indicators (for context/risk)
        avg_turnover_egp: 20-day average turnover from liquidity gate
        latest_turnover_egp: Today's turnover
        stock_change_20d: Stock's % change over 20 trading days
        egx30_change_20d: EGX 30 index % change over 20 trading days
        confirmation: ConfirmationResult from confirmation_check()
        rr_ratio: Risk/Reward ratio from risk_filter()
        stop_loss: Stop-loss price
        target: Target (resistance) price
        data_freshness: "live", "fresh", "stale", or "unknown"
        data_quality: 0.0 to 1.0

    Returns:
        ScoringResult with total score, recommendation, and factors.
    """
    # Get EMA50 and RSI from indicators
    ema50_ind = _get_indicator(analysis, "EMA 50")
    ema50 = ema50_ind.value if ema50_ind else 0.0

    rsi_ind = _get_indicator(analysis, "RSI")
    rsi = rsi_ind.value if rsi_ind else 50.0

    # Compute all 4 factors
    factors = [
        _score_liquidity_strength(avg_turnover_egp, latest_turnover_egp),
        _score_trend_rs(analysis.current_price, ema50, rsi, stock_change_20d, egx30_change_20d),
        _score_confirmation(confirmation),
        _score_risk_reward(rr_ratio),
    ]

    # Total score
    total = sum(f.weighted_score for f in factors)
    total = max(0, min(100, round(total)))

    # Risk assessment
    risk_level, risk_reason = _assess_risk(analysis, rr_ratio)

    # Determine recommendation
    if data_quality < config.MIN_DATA_QUALITY_WATCH:
        recommendation = "No Trade"
        reason = "جودة البيانات غير كافية"
    elif data_freshness == "stale":
        recommendation = "No Trade"
        reason = "البيانات قديمة"
    elif total >= config.BUY_THRESHOLD:
        recommendation = "Buy"
        reason = f"درجة {total}/100 — سيولة قوية + اتجاه مؤكد"
    elif total >= config.WATCH_THRESHOLD:
        recommendation = "Watch"
        reason = f"درجة {total}/100 — إيجابي ولكن يحتاج تأكيد إضافي"
    elif total <= config.SELL_THRESHOLD:
        recommendation = "Sell"
        reason = f"درجة {total}/100 — إشارات هابطة"
    else:
        recommendation = "No Trade"
        reason = f"درجة {total}/100 — إشارات متضاربة"

    rec_ar = {
        "Buy": "شراء 🟢", "Watch": "مراقبة 🟡",
        "Sell": "بيع 🔴", "No Trade": "لا تداول ⚪",
    }.get(recommendation, "لا تداول ⚪")

    # Collect pass/fail reasons
    pass_reasons = [f"{f.name_ar}: {f.reason}" for f in factors if f.raw_score >= 60]
    fail_reasons = [f"{f.name_ar}: {f.reason}" for f in factors if f.raw_score <= 40]

    return ScoringResult(
        total_score=total,
        recommendation=recommendation,
        recommendation_ar=rec_ar,
        factors=factors,
        risk_level=risk_level,
        risk_reason=risk_reason,
        data_quality=data_quality,
        data_freshness=data_freshness,
        pass_reasons=pass_reasons,
        fail_reasons=fail_reasons,
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        rr_ratio=round(rr_ratio, 2),
        surge_info=confirmation.details.get("today_vol", ""),
        confirmation_info=confirmation.confirmation_type,
    )


# ─── Backward Compatibility ──────────────────────────────────────────────────

def compute_score(
    analysis: StockAnalysis,
    data_freshness: str = "fresh",
    data_quality: float = 1.0,
) -> ScoringResult:
    """
    Legacy scoring function (v1) — kept for backward compatibility.

    New code should use compute_score_v2() with the Liquidity-First pipeline.
    This function computes the old 8-factor score for stocks that haven't
    been through the v2 filter pipeline.
    """
    factors = [
        _score_trend_v1(analysis),
        _score_momentum_v1(analysis),
        _score_volume_v1(analysis),
        _score_volatility_v1(analysis),
        _score_breakout_v1(analysis),
        _score_risk_reward_v1(analysis),
        _score_data_freshness_v1(data_freshness, data_quality),
        _score_signal_alignment_v1(analysis),
    ]

    total = sum(f.weighted_score for f in factors)
    total = max(0, min(100, round(total)))

    risk_level, risk_reason = _assess_risk(analysis, analysis.risk_reward_ratio)

    if data_quality < 0.7:
        recommendation = "No Trade"
    elif data_freshness == "stale":
        recommendation = "No Trade"
    elif total >= 70:
        recommendation = "Buy"
    elif total >= 50:
        recommendation = "Watch"
    elif total <= 30:
        recommendation = "Sell"
    else:
        recommendation = "No Trade"

    rec_ar = {
        "Buy": "شراء 🟢", "Watch": "مراقبة 🟡",
        "Sell": "بيع 🔴", "No Trade": "لا تداول ⚪",
    }.get(recommendation, "لا تداول ⚪")

    pass_reasons = [f"{f.name_ar}: {f.reason}" for f in factors if f.raw_score >= 60]
    fail_reasons = [f"{f.name_ar}: {f.reason}" for f in factors if f.raw_score <= 40]

    return ScoringResult(
        total_score=total,
        recommendation=recommendation,
        recommendation_ar=rec_ar,
        factors=factors,
        risk_level=risk_level,
        risk_reason=risk_reason,
        data_quality=data_quality,
        data_freshness=data_freshness,
        pass_reasons=pass_reasons,
        fail_reasons=fail_reasons,
    )


# ─── v1 Factor Functions (legacy, kept for backward compatibility) ───────────

WEIGHTS_V1 = {
    "trend": 20, "momentum": 15, "volume": 12, "volatility": 8,
    "breakout": 15, "risk_reward": 10, "data_freshness": 10, "signal_alignment": 10,
}


def _score_trend_v1(analysis: StockAnalysis) -> ScoreFactor:
    sma = _get_indicator(analysis, "SMA Trend")
    adx = _get_indicator(analysis, "ADX")
    st = _get_indicator(analysis, "SuperTrend")
    score = 50.0
    reasons = []
    if sma:
        if sma.signal == 1: score += 15; reasons.append("الاتجاه صاعد (SMA)")
        elif sma.signal == -1: score -= 15; reasons.append("الاتجاه هابط (SMA)")
        if "Golden Cross" in sma.note or "تقاطع ذهبي" in sma.note: score += 10; reasons.append("تقاطع ذهبي")
        elif "Death Cross" in sma.note or "تقاطع الموت" in sma.note: score -= 10; reasons.append("تقاطع الموت")
    if adx:
        if adx.value > 25: score += 10 if adx.signal == 1 else -10; reasons.append(f"اتجاه قوي (ADX={adx.value:.0f})")
        elif adx.value < 20: score -= 5; reasons.append("اتجاه ضعيف")
    if st:
        if st.signal == 1: score += 10; reasons.append("SuperTrend صاعد")
        elif st.signal == -1: score -= 10; reasons.append("SuperTrend هابط")
    score = max(0, min(100, score))
    return ScoreFactor("Trend Quality", "جودة الاتجاه", score, WEIGHTS_V1["trend"], score * WEIGHTS_V1["trend"] / 100, "، ".join(reasons) if reasons else "اتجاه محايد")


def _score_momentum_v1(analysis: StockAnalysis) -> ScoreFactor:
    rsi = _get_indicator(analysis, "RSI")
    stoch = _get_indicator(analysis, "Stochastic")
    macd = _get_indicator(analysis, "MACD")
    stoch_rsi = _get_indicator(analysis, "Stochastic RSI")
    score = 50.0
    reasons = []
    if rsi:
        if rsi.signal == 1: score += 10; reasons.append(f"RSI={rsi.value:.0f} (زخم صاعد)")
        elif rsi.signal == -1: score -= 10; reasons.append(f"RSI={rsi.value:.0f} (زخم هابط)")
    if stoch:
        if stoch.signal == 1: score += 8; reasons.append("استوكاستك صاعد")
        elif stoch.signal == -1: score -= 8; reasons.append("استوكاستك هابط")
    if macd:
        if macd.signal == 1: score += 10; reasons.append("MACD صاعد")
        elif macd.signal == -1: score -= 10; reasons.append("MACD هابط")
    if stoch_rsi:
        if stoch_rsi.signal == 1: score += 7; reasons.append("Stochastic RSI صاعد")
        elif stoch_rsi.signal == -1: score -= 7; reasons.append("Stochastic RSI هابط")
    score = max(0, min(100, score))
    return ScoreFactor("Momentum", "الزخم", score, WEIGHTS_V1["momentum"], score * WEIGHTS_V1["momentum"] / 100, "، ".join(reasons) if reasons else "زخم محايد")


def _score_volume_v1(analysis: StockAnalysis) -> ScoreFactor:
    obv = _get_indicator(analysis, "OBV")
    vol_ratio = _get_indicator(analysis, "Volume Ratio")
    score = 50.0
    reasons = []
    if obv:
        if obv.signal == 1: score += 15; reasons.append("تراكم حجمي إيجابي")
        elif obv.signal == -1: score -= 15; reasons.append("توزيع حجمي سلبي")
    if vol_ratio:
        if vol_ratio.value > 1.5: score += 10; reasons.append(f"حجم مرتفع ({vol_ratio.value:.1f}x)")
        elif vol_ratio.value < 0.5: score -= 5; reasons.append(f"حجم منخفض ({vol_ratio.value:.1f}x)")
    score = max(0, min(100, score))
    return ScoreFactor("Volume Confirmation", "تأكيد الحجم", score, WEIGHTS_V1["volume"], score * WEIGHTS_V1["volume"] / 100, "، ".join(reasons) if reasons else "حجم طبيعي")


def _score_volatility_v1(analysis: StockAnalysis) -> ScoreFactor:
    atr = _get_indicator(analysis, "ATR")
    bb = _get_indicator(analysis, "Bollinger")
    score = 50.0
    reasons = []
    if atr:
        if atr.value < 2: score += 5; reasons.append("تذبذب منخفض")
        elif atr.value > 5: score -= 10; reasons.append("تذبذب مرتفع")
        else: reasons.append("تذبذب متوسط")
    if bb:
        if bb.signal == 1: score += 8; reasons.append("السعر قرب الحد السفلي للبولينجر")
        elif bb.signal == -1: score -= 5; reasons.append("السعر قرب الحد العلوي للبولينجر")
    score = max(0, min(100, score))
    return ScoreFactor("Volatility", "التذبذب", score, WEIGHTS_V1["volatility"], score * WEIGHTS_V1["volatility"] / 100, "، ".join(reasons) if reasons else "تذبذب طبيعي")


def _score_breakout_v1(analysis: StockAnalysis) -> ScoreFactor:
    breakout = _get_indicator(analysis, "Breakout")
    st = _get_indicator(analysis, "SuperTrend")
    score = 50.0
    reasons = []
    if breakout:
        if breakout.signal == 1: score += 25; reasons.append("اختراق مقاومة")
        elif breakout.signal == -1: score -= 25; reasons.append("كسر دعم")
        else: reasons.append("لا يوجد اختراق")
    if st:
        if st.signal == 1: score += 10; reasons.append("تأكيد SuperTrend صاعد")
        elif st.signal == -1: score -= 10; reasons.append("تأكيد SuperTrend هابط")
    score = max(0, min(100, score))
    return ScoreFactor("Breakout", "الاختراق", score, WEIGHTS_V1["breakout"], score * WEIGHTS_V1["breakout"] / 100, "، ".join(reasons) if reasons else "لا يوجد اختراق واضح")


def _score_risk_reward_v1(analysis: StockAnalysis) -> ScoreFactor:
    rr = _get_indicator(analysis, "Risk/Reward")
    score = 50.0
    reasons = []
    if rr:
        ratio = rr.value
        if ratio >= 3.0: score = 90; reasons.append(f"نسبة ممتازة ({ratio:.1f}:1)")
        elif ratio >= 2.0: score = 75; reasons.append(f"نسبة جيدة ({ratio:.1f}:1)")
        elif ratio >= 1.0: score = 55; reasons.append(f"نسبة مقبولة ({ratio:.1f}:1)")
        elif ratio > 0: score = 30; reasons.append(f"نسبة ضعيفة ({ratio:.1f}:1)")
        else: score = 50; reasons.append("لا تتوفر مستويات واضحة")
    else:
        reasons.append("لا تتوفر بيانات كافية")
    return ScoreFactor("Risk/Reward", "المخاطرة/العائد", score, WEIGHTS_V1["risk_reward"], score * WEIGHTS_V1["risk_reward"] / 100, "، ".join(reasons))


def _score_data_freshness_v1(freshness: str, data_quality: float) -> ScoreFactor:
    if freshness == "live": score = 95
    elif freshness == "fresh": score = 80
    elif freshness == "stale": score = 30
    else: score = 50
    if data_quality < 0.7: score = min(score, 40)
    return ScoreFactor("Data Freshness", "جودة البيانات", score, WEIGHTS_V1["data_freshness"], score * WEIGHTS_V1["data_freshness"] / 100, f"{freshness} (جودة {data_quality:.0%})")


def _score_signal_alignment_v1(analysis: StockAnalysis) -> ScoreFactor:
    bullish = sum(1 for ind in analysis.indicators if ind.signal == 1)
    bearish = sum(1 for ind in analysis.indicators if ind.signal == -1)
    total_signals = bullish + bearish
    if total_signals == 0: score = 50; reason = "إشارات متعادلة"
    else:
        bull_pct = bullish / total_signals * 100
        score = bull_pct
        reason = f"{bullish} صعودي / {bearish} هابطي ({bull_pct:.0f}% صعودي)"
    return ScoreFactor("Signal Alignment", "توافق الإشارات", score, WEIGHTS_V1["signal_alignment"], score * WEIGHTS_V1["signal_alignment"] / 100, reason)
