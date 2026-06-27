"""
scoring.py
----------
Deterministic scoring engine (0-100) for EGX stock analysis.

The score is based ONLY on computed technical signals and verified market data.
Every score is fully explainable — each factor contributes a weighted sub-score.

Recommendations:
  Buy      — score >= 70 AND data_quality >= 0.8
  Watch    — score 50-69 AND data_quality >= 0.7
  Sell     — score <= 30 AND data_quality >= 0.8
  No Trade — everything else (weak setup, poor data, or neutral)

No AI is involved in scoring. AI only explains the computed results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from indicators import StockAnalysis, IndicatorResult

logger = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

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
    total_score: int                      # 0-100 rounded
    recommendation: str                   # "Buy", "Watch", "Sell", "No Trade"
    recommendation_ar: str                # Arabic recommendation
    factors: list[ScoreFactor] = field(default_factory=list)
    risk_level: str = "Medium"            # "Low", "Medium", "High"
    risk_reason: str = ""
    data_quality: float = 1.0             # 0.0-1.0
    data_freshness: str = "fresh"
    pass_reasons: list[str] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.recommendation in ("Buy", "Sell")


# ─── Factor Weights (must sum to 100) ────────────────────────────────────────

WEIGHTS = {
    "trend": 20,
    "momentum": 15,
    "volume": 12,
    "volatility": 8,
    "breakout": 15,
    "risk_reward": 10,
    "data_freshness": 10,
    "signal_alignment": 10,
}

assert sum(WEIGHTS.values()) == 100, "Factor weights must sum to 100"


# ─── Scoring Functions ───────────────────────────────────────────────────────

def _get_indicator(analysis: StockAnalysis, name: str) -> Optional[IndicatorResult]:
    """Find an indicator by name."""
    for ind in analysis.indicators:
        if ind.name == name:
            return ind
    return None


def _score_trend(analysis: StockAnalysis) -> ScoreFactor:
    """Trend quality: SMA alignment, ADX, SuperTrend."""
    sma = _get_indicator(analysis, "SMA Trend")
    adx = _get_indicator(analysis, "ADX")
    st = _get_indicator(analysis, "SuperTrend")

    score = 50.0  # Neutral start
    reasons = []

    if sma:
        if sma.signal == 1:
            score += 15
            reasons.append("الاتجاه صاعد (SSA)")
        elif sma.signal == -1:
            score -= 15
            reasons.append("الاتجاه هابط (SMA)")
        if "Golden Cross" in sma.note:
            score += 10
            reasons.append("تقاطع ذهبي")
        elif "Death Cross" in sma.note:
            score -= 10
            reasons.append("تقاطع الموت")

    if adx:
        if adx.value > 25:
            score += 10 if adx.signal == 1 else -10
            reasons.append(f"اتجاه قوي (ADX={adx.value:.0f})")
        elif adx.value < 20:
            score -= 5
            reasons.append("اتجاه ضعيف")

    if st:
        if st.signal == 1:
            score += 10
            reasons.append("SuperTrend صاعد")
        elif st.signal == -1:
            score -= 10
            reasons.append("SuperTrend هابط")

    score = max(0, min(100, score))
    return ScoreFactor(
        name="Trend Quality", name_ar="جودة الاتجاه",
        raw_score=score, weight=WEIGHTS["trend"],
        weighted_score=score * WEIGHTS["trend"] / 100,
        reason="، ".join(reasons) if reasons else "اتجاه محايد",
    )


def _score_momentum(analysis: StockAnalysis) -> ScoreFactor:
    """Momentum: RSI, Stochastic, MACD, Stochastic RSI."""
    rsi = _get_indicator(analysis, "RSI")
    stoch = _get_indicator(analysis, "Stochastic")
    macd = _get_indicator(analysis, "MACD")
    stoch_rsi = _get_indicator(analysis, "Stochastic RSI")

    score = 50.0
    reasons = []

    if rsi:
        if rsi.signal == 1:
            score += 10
            reasons.append(f"RSI={rsi.value:.0f} (زخم صاعد)")
        elif rsi.signal == -1:
            score -= 10
            reasons.append(f"RSI={rsi.value:.0f} (زخم هابط)")

    if stoch:
        if stoch.signal == 1:
            score += 8
            reasons.append("استوكاستك صاعد")
        elif stoch.signal == -1:
            score -= 8
            reasons.append("استوكاستك هابط")

    if macd:
        if macd.signal == 1:
            score += 10
            reasons.append("MACD صاعد")
        elif macd.signal == -1:
            score -= 10
            reasons.append("MACD هابط")

    if stoch_rsi:
        if stoch_rsi.signal == 1:
            score += 7
            reasons.append("Stochastic RSI صاعد")
        elif stoch_rsi.signal == -1:
            score -= 7
            reasons.append("Stochastic RSI هابط")

    score = max(0, min(100, score))
    return ScoreFactor(
        name="Momentum", name_ar="الزخم",
        raw_score=score, weight=WEIGHTS["momentum"],
        weighted_score=score * WEIGHTS["momentum"] / 100,
        reason="، ".join(reasons) if reasons else "زخم محايد",
    )


def _score_volume(analysis: StockAnalysis) -> ScoreFactor:
    """Volume confirmation: OBV, volume ratio."""
    obv = _get_indicator(analysis, "OBV")
    vol_ratio = _get_indicator(analysis, "Volume Ratio")

    score = 50.0
    reasons = []

    if obv:
        if obv.signal == 1:
            score += 15
            reasons.append("تراكم حجمي إيجابي")
        elif obv.signal == -1:
            score -= 15
            reasons.append("توزيع حجمي سلبي")

    if vol_ratio:
        if vol_ratio.value > 1.5:
            score += 10
            reasons.append(f"حجم تداول مرتفع ({vol_ratio.value:.1f}x)")
        elif vol_ratio.value < 0.5:
            score -= 5
            reasons.append(f"حجم تداول منخفض ({vol_ratio.value:.1f}x)")

    score = max(0, min(100, score))
    return ScoreFactor(
        name="Volume Confirmation", name_ar="تأكيد الحجم",
        raw_score=score, weight=WEIGHTS["volume"],
        weighted_score=score * WEIGHTS["volume"] / 100,
        reason="، ".join(reasons) if reasons else "حجم طبيعي",
    )


def _score_volatility(analysis: StockAnalysis) -> ScoreFactor:
    """Volatility condition: ATR, Bollinger Bands."""
    atr = _get_indicator(analysis, "ATR")
    bb = _get_indicator(analysis, "Bollinger")

    score = 50.0
    reasons = []

    if atr:
        # Moderate volatility is good for trading; extreme is risky
        if atr.value < 2:
            score += 5
            reasons.append("تذبذب منخفض (استقرار)")
        elif atr.value > 5:
            score -= 10
            reasons.append("تذبذب مرتفع (مخاطرة)")
        else:
            reasons.append("تذبذب متوسط")

    if bb:
        if bb.signal == 1:
            score += 8
            reasons.append("السعر قرب الحد السفلي للبولينجر")
        elif bb.signal == -1:
            score -= 5
            reasons.append("السعر قرب الحد العلوي للبولينجر")

    score = max(0, min(100, score))
    return ScoreFactor(
        name="Volatility", name_ar="التذبذب",
        raw_score=score, weight=WEIGHTS["volatility"],
        weighted_score=score * WEIGHTS["volatility"] / 100,
        reason="، ".join(reasons) if reasons else "تذبذب طبيعي",
    )


def _score_breakout(analysis: StockAnalysis) -> ScoreFactor:
    """Breakout strength: support/resistance breakout, SuperTrend."""
    breakout = _get_indicator(analysis, "Breakout")
    st = _get_indicator(analysis, "SuperTrend")

    score = 50.0
    reasons = []

    if breakout:
        if "bullish" in breakout.note.lower():
            score += 25
            reasons.append("اختراق مقاومة")
        elif "bearish" in breakout.note.lower():
            score -= 25
            reasons.append("كسر دعم")
        else:
            reasons.append("لا يوجد اختراق")

    if st:
        if st.signal == 1:
            score += 10
            reasons.append("تأكيد SuperTrend صاعد")
        elif st.signal == -1:
            score -= 10
            reasons.append("تأكيد SuperTrend هابط")

    score = max(0, min(100, score))
    return ScoreFactor(
        name="Breakout", name_ar="الاختراق",
        raw_score=score, weight=WEIGHTS["breakout"],
        weighted_score=score * WEIGHTS["breakout"] / 100,
        reason="، ".join(reasons) if reasons else "لا يوجد اختراق واضح",
    )


def _score_risk_reward(analysis: StockAnalysis) -> ScoreFactor:
    """Risk/reward quality based on support/resistance levels."""
    rr = _get_indicator(analysis, "Risk/Reward")

    score = 50.0
    reasons = []

    if rr:
        ratio = rr.value
        if ratio >= 3.0:
            score = 90
            reasons.append(f"نسبة مخاطرة/عائد ممتازة ({ratio:.1f}:1)")
        elif ratio >= 2.0:
            score = 75
            reasons.append(f"نسبة مخاطرة/عائد جيدة ({ratio:.1f}:1)")
        elif ratio >= 1.0:
            score = 55
            reasons.append(f"نسبة مخاطرة/عائد مقبولة ({ratio:.1f}:1)")
        elif ratio > 0:
            score = 30
            reasons.append(f"نسبة مخاطرة/عائد ضعيفة ({ratio:.1f}:1)")
        else:
            score = 50
            reasons.append("لا تتوفر مستويات واضحة")
    else:
        reasons.append("لا تتوفر بيانات كافية")

    return ScoreFactor(
        name="Risk/Reward", name_ar="المخاطرة/العائد",
        raw_score=score, weight=WEIGHTS["risk_reward"],
        weighted_score=score * WEIGHTS["risk_reward"] / 100,
        reason="، ".join(reasons),
    )


def _score_data_freshness(freshness: str, data_quality: float) -> ScoreFactor:
    """Data freshness score."""
    scores = {"live": 100, "fresh": 80, "stale": 30, "unknown": 10}
    score = scores.get(freshness, 50)
    score = score * data_quality  # Scale by data quality

    labels = {"live": "بيانات حية", "fresh": "بيانات حديثة",
              "stale": "بيانات قديمة", "unknown": "بيانات غير مؤكدة"}
    label = labels.get(freshness, "بيانات غير مؤكدة")

    return ScoreFactor(
        name="Data Freshness", name_ar="حداثة البيانات",
        raw_score=score, weight=WEIGHTS["data_freshness"],
        weighted_score=score * WEIGHTS["data_freshness"] / 100,
        reason=label,
    )


def _score_signal_alignment(analysis: StockAnalysis) -> ScoreFactor:
    """How many directional indicators agree."""
    directional = [i for i in analysis.indicators if i.signal != 0 and i.name != "ATR"]
    if not directional:
        return ScoreFactor(
            name="Signal Alignment", name_ar="توافق الإشارات",
            raw_score=50, weight=WEIGHTS["signal_alignment"],
            weighted_score=50 * WEIGHTS["signal_alignment"] / 100,
            reason="لا توجد إشارات واضحة",
        )

    bullish = sum(1 for i in directional if i.signal > 0)
    bearish = sum(1 for i in directional if i.signal < 0)
    total = len(directional)

    # Score: 100 if all bullish, 0 if all bearish, 50 if balanced
    if total > 0:
        score = 50 + (bullish - bearish) / total * 50
    else:
        score = 50

    score = max(0, min(100, score))

    if bullish > bearish:
        reason = f"{bullish}/{total} إشارة صاعدة"
    elif bearish > bullish:
        reason = f"{bearish}/{total} إشارة هابطة"
    else:
        reason = f"إشارات متوازنة ({bullish} صاعد / {bearish} هابط)"

    return ScoreFactor(
        name="Signal Alignment", name_ar="توافق الإشارات",
        raw_score=score, weight=WEIGHTS["signal_alignment"],
        weighted_score=score * WEIGHTS["signal_alignment"] / 100,
        reason=reason,
    )


# ─── Risk Assessment ─────────────────────────────────────────────────────────

def _assess_risk(analysis: StockAnalysis) -> tuple[str, str]:
    """Determine risk level and reason."""
    atr = _get_indicator(analysis, "ATR")
    bb = _get_indicator(analysis, "Bollinger")
    reasons = []

    risk = "Medium"

    if atr:
        if atr.value > 5:
            risk = "High"
            reasons.append(f"تذبذب مرتفع ({atr.value:.1f}%)")
        elif atr.value < 2:
            risk = "Low"
            reasons.append(f"تذبذب منخفض ({atr.value:.1f}%)")

    if bb:
        if bb.signal == -1:
            reasons.append("السعر قرب الحد العلوي للبولينجر")

    # Check if near resistance (rejection risk)
    if analysis.resistance > 0 and analysis.current_price > 0:
        dist_to_resistance = (analysis.resistance - analysis.current_price) / analysis.current_price * 100
        if dist_to_resistance < 2:
            reasons.append(f"قرب مقاومة ({dist_to_resistance:.1f}%)")
            if risk == "Low":
                risk = "Medium"

    if not reasons:
        reasons.append("مخاطرة معتدلة")

    return risk, "، ".join(reasons)


# ─── Main Scoring Function ───────────────────────────────────────────────────

def compute_score(
    analysis: StockAnalysis,
    data_freshness: str = "fresh",
    data_quality: float = 1.0,
) -> ScoringResult:
    """
    Compute a deterministic 0-100 score from technical analysis.
    Every factor is computed from real indicator data — no AI, no guessing.

    Args:
        analysis: StockAnalysis with computed indicators
        data_freshness: "live", "fresh", "stale", or "unknown"
        data_quality: 0.0 to 1.0

    Returns:
        ScoringResult with total score, recommendation, and explainable factors.
    """
    # Compute all factors
    factors = [
        _score_trend(analysis),
        _score_momentum(analysis),
        _score_volume(analysis),
        _score_volatility(analysis),
        _score_breakout(analysis),
        _score_risk_reward(analysis),
        _score_data_freshness(data_freshness, data_quality),
        _score_signal_alignment(analysis),
    ]

    # Total score = sum of weighted scores
    total = sum(f.weighted_score for f in factors)
    total = max(0, min(100, round(total)))

    # Risk assessment
    risk_level, risk_reason = _assess_risk(analysis)

    # Determine recommendation
    if data_quality < 0.7:
        recommendation = "No Trade"
        reason = "جودة البيانات غير كافية"
    elif data_freshness == "stale":
        recommendation = "No Trade"
        reason = "البيانات قديمة"
    elif total >= 70:
        recommendation = "Buy"
        reason = f"درجة {total}/100 — إشارات صاعدة قوية"
    elif total >= 50:
        recommendation = "Watch"
        reason = f"درجة {total}/100 — إشارات إيجابية ولكن غير مؤكدة"
    elif total <= 30:
        recommendation = "Sell"
        reason = f"درجة {total}/100 — إشارات هابطة قوية"
    else:
        recommendation = "No Trade"
        reason = f"درجة {total}/100 — إشارات متضاربة"

    # Arabic recommendation
    rec_ar = {
        "Buy": "شراء 🟢",
        "Watch": "مراقبة 🟡",
        "Sell": "بيع 🔴",
        "No Trade": "لا تداول ⚪",
    }.get(recommendation, "لا تداول ⚪")

    # Collect pass/fail reasons
    pass_reasons = []
    fail_reasons = []
    for f in factors:
        if f.raw_score >= 60:
            pass_reasons.append(f"{f.name_ar}: {f.reason}")
        elif f.raw_score <= 40:
            fail_reasons.append(f"{f.name_ar}: {f.reason}")

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
