"""
indicators.py
-------------
Professional technical indicators for EGX stock analysis.

Indicators calculated (15+):
  - RSI (14)
  - Stochastic Oscillator (14, 3, 3)
  - Stochastic RSI (14)
  - MACD (12, 26, 9)
  - Bollinger Bands (20, 2)
  - SMA (20, 50, 200)
  - EMA (20, 50, 200)
  - ADX (14) — trend strength
  - OBV (On-Balance Volume)
  - Volume Profile (POC, Value Area)
  - ATR (14) — volatility
  - Williams %R (14)
  - VWAP (Volume Weighted Average Price)
  - SuperTrend (10, 3.0)
  - Support and Resistance
  - Breakout Detection
  - Risk/Reward Ratio
  - Volume Ratio (current vs average)

Each indicator returns a signal: bullish (+1), bearish (-1), or neutral (0).
The StockAnalysis dataclass aggregates all signals into a composite score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class IndicatorResult:
    """Result of a single technical indicator."""
    name: str
    name_ar: str
    value: float
    signal: int          # +1 bullish, -1 bearish, 0 neutral
    signal_text: str     # "صاعد", "هابط", "محايد"
    note: str = ""


@dataclass
class VolumeProfileResult:
    """Simplified Volume Profile analysis."""
    poc: float
    value_area_high: float
    value_area_low: float
    current_price_position: str
    signal: int
    signal_text: str


@dataclass
class SupportResistanceResult:
    """Support and resistance levels."""
    support: float
    resistance: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float


@dataclass
class StockAnalysis:
    """Complete technical analysis for a single stock."""
    ticker: str
    name: str
    name_ar: str
    current_price: float
    daily_change_pct: float
    volume: int
    indicators: list[IndicatorResult] = field(default_factory=list)
    volume_profile: Optional[VolumeProfileResult] = None
    support_resistance: Optional[SupportResistanceResult] = None
    composite_score: int = 0
    signal_label: str = "محايد 🟡"
    signal_score_pct: float = 0.0
    bullish_reasons: list[str] = field(default_factory=list)
    bearish_reasons: list[str] = field(default_factory=list)
    # Extended fields for scoring engine
    data_freshness: str = "unknown"
    data_quality: float = 1.0
    timestamp: str = ""
    support: float = 0.0
    resistance: float = 0.0
    risk_reward_ratio: float = 0.0
    vwap: float = 0.0
    supertrend_signal: str = "unknown"
    breakout_type: str = "none"

    @property
    def is_bullish(self) -> bool:
        return self.composite_score >= 2

    @property
    def is_strong_bullish(self) -> bool:
        return self.composite_score >= 4


# ─── Helper ──────────────────────────────────────────────────────────────────

def _signal_text(score: int) -> str:
    if score > 0:
        return "صاعد"
    elif score < 0:
        return "هابط"
    return "محايد"


# ─── Core Indicators ─────────────────────────────────────────────────────────

def calc_rsi(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """RSI — oversold < 30 (bullish reversal), overbought > 70 (bearish correction)."""
    rsi_ind = RSIIndicator(close=df["Close"], window=period)
    rsi = rsi_ind.rsi().iloc[-1]

    if rsi < 30:
        signal, note = 1, "تشبع بيعي — احتمال ارتداد صاعد"
    elif rsi > 70:
        signal, note = -1, "تشبع شرائي — احتمال تصحيح هابط"
    elif rsi > 55:
        signal, note = 1, "زخم صاعد قوي"
    elif rsi < 45:
        signal, note = -1, "زخم هابط ضعيف"
    else:
        signal, note = 0, "منطقة محايدة"

    return IndicatorResult("RSI", "مؤشر القوة النسبية", round(float(rsi), 2), signal, _signal_text(signal), note)


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult:
    """Stochastic Oscillator — %K and %D crossover."""
    stoch = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window=k_period, smooth_window=d_period,
    )
    k_val = stoch.stoch().iloc[-1]
    d_val = stoch.stoch_signal().iloc[-1]

    k_prev = stoch.stoch().iloc[-2] if len(stoch.stoch()) > 1 else k_val
    d_prev = stoch.stoch_signal().iloc[-2] if len(stoch.stoch_signal()) > 1 else d_val

    bullish_cross = k_prev <= d_prev and k_val > d_val
    bearish_cross = k_prev >= d_prev and k_val < d_val

    if k_val < 20 and bullish_cross:
        signal, note = 1, f"تقاطع صاعد في تشبع بيعي (%K={k_val:.1f})"
    elif k_val > 80 and bearish_cross:
        signal, note = -1, f"تقاطع هابط في تشبع شرائي (%K={k_val:.1f})"
    elif k_val < 20:
        signal, note = 1, "منطقة تشبع بيعي"
    elif k_val > 80:
        signal, note = -1, "منطقة تشبع شرائي"
    elif bullish_cross:
        signal, note = 0, "تقاطع صاعد في منطقة محايدة"
    elif bearish_cross:
        signal, note = 0, "تقاطع هابط في منطقة محايدة"
    else:
        signal, note = 0, "منطقة محايدة"

    return IndicatorResult("Stochastic", "مؤشر الاستوكاستك", round(float(k_val), 2), signal, _signal_text(signal), note)


def calc_stochastic_rsi(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """Stochastic RSI — RSI of RSI. More sensitive than RSI alone."""
    rsi_ind = RSIIndicator(close=df["Close"], window=period)
    rsi_series = rsi_ind.rsi().dropna()

    if len(rsi_series) < period:
        return IndicatorResult("Stochastic RSI", "ستوكاستك RSI", 50.0, 0, "محايد", "بيانات غير كافية")

    # Stochastic of RSI
    min_rsi = rsi_series.rolling(window=period).min()
    max_rsi = rsi_series.rolling(window=period).max()
    stoch_rsi = (rsi_series - min_rsi) / (max_rsi - min_rsi) * 100
    stoch_rsi = stoch_rsi.dropna()

    if len(stoch_rsi) == 0:
        return IndicatorResult("Stochastic RSI", "ستوكاستك RSI", 50.0, 0, "محايد", "بيانات غير كافية")

    val = stoch_rsi.iloc[-1]

    if val < 20:
        signal, note = 1, "تشبع بيعي شديد"
    elif val > 80:
        signal, note = -1, "تشبع شرائي شديد"
    elif val < 40:
        signal, note = 0, "منطقة دنيا"
    elif val > 60:
        signal, note = 0, "منطقة عليا"
    else:
        signal, note = 0, "منطقة محايدة"

    return IndicatorResult("Stochastic RSI", "ستوكاستك RSI", round(float(val), 2), signal, _signal_text(signal), note)


def calc_macd(df: pd.DataFrame) -> IndicatorResult:
    """MACD — trend momentum via 12/26 EMA crossover."""
    macd_ind = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd().iloc[-1]
    signal_line = macd_ind.macd_signal().iloc[-1]
    histogram = macd_ind.macd_diff().iloc[-1]

    macd_prev = macd_ind.macd().iloc[-2] if len(macd_ind.macd()) > 1 else macd_line
    signal_prev = macd_ind.macd_signal().iloc[-2] if len(macd_ind.macd_signal()) > 1 else signal_line

    bullish_cross = macd_prev <= signal_prev and macd_line > signal_line
    bearish_cross = macd_prev >= signal_prev and macd_line < signal_line

    if bullish_cross and histogram > 0:
        signal, note = 1, "تقاطع صاعد — زخم إيجابي قوي"
    elif bearish_cross and histogram < 0:
        signal, note = -1, "تقاطع هابط — زخم سلبي قوي"
    elif macd_line > signal_line:
        signal, note = 1, "MACD فوق خط الإشارة"
    elif macd_line < signal_line:
        signal, note = -1, "MACD تحت خط الإشارة"
    else:
        signal, note = 0, "زخم محايد"

    return IndicatorResult("MACD", "ماكد", round(float(macd_line), 4), signal, _signal_text(signal), note)


def calc_bollinger(df: pd.DataFrame, window: int = 20, dev: int = 2) -> IndicatorResult:
    """Bollinger Bands — price relative to bands."""
    bb = BollingerBands(close=df["Close"], window=window, window_dev=dev)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    price = df["Close"].iloc[-1]

    band_width = upper - lower
    if band_width > 0:
        position = (price - lower) / band_width
    else:
        position = 0.5

    if price <= lower * 1.01:
        signal, note = 1, "السعر عند الحد السفلي — احتمال ارتداد"
    elif price >= upper * 0.99:
        signal, note = -1, "السعر عند الحد العلوي — احتمال تصحيح"
    elif position < 0.3:
        signal, note = 1, "السعر قرب الحد السفلي"
    elif position > 0.7:
        signal, note = -1, "السعر قرب الحد العلوي"
    else:
        signal, note = 0, "السعر داخل النطاق المتوسط"

    return IndicatorResult("Bollinger", "بولينجر باند", round(float(position * 100), 1), signal, _signal_text(signal), note)


def calc_sma_trend(df: pd.DataFrame) -> IndicatorResult:
    """SMA 20/50/200 trend analysis — golden cross / death cross."""
    sma20 = SMAIndicator(close=df["Close"], window=20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(close=df["Close"], window=50).sma_indicator().iloc[-1]
    price = df["Close"].iloc[-1]

    has_sma200 = len(df) >= 200
    sma200 = SMAIndicator(close=df["Close"], window=200).sma_indicator().iloc[-1] if has_sma200 else None

    bullish_stack = price > sma20 > sma50
    bearish_stack = price < sma20 < sma50
    golden_cross = has_sma200 and sma50 > sma200
    death_cross = has_sma200 and sma50 < sma200

    if bullish_stack and golden_cross:
        signal, note = 1, "ترتيب صاعد كامل (Golden Cross) — اتجاه صاعد قوي"
    elif bullish_stack:
        signal, note = 1, "السعر فوق SMA20 و SMA50 — اتجاه صاعد"
    elif bearish_stack and death_cross:
        signal, note = -1, "ترتيب هابط كامل (Death Cross) — اتجاه هابط قوي"
    elif bearish_stack:
        signal, note = -1, "السعر تحت SMA20 و SMA50 — اتجاه هابط"
    elif price > sma50:
        signal, note = 1, "السعر فوق SMA50 — اتجاه إيجابي نسبيًا"
    elif price < sma50:
        signal, note = -1, "السعر تحت SMA50 — اتجاه سلبي نسبيًا"
    else:
        signal, note = 0, "اتجاه متذبذب"

    return IndicatorResult("SMA Trend", "المتوسطات المتحركة", round(float(sma20), 2), signal, _signal_text(signal), note)


def calc_ema_trend(df: pd.DataFrame) -> IndicatorResult:
    """EMA 20/50/200 — more responsive than SMA."""
    ema20 = EMAIndicator(close=df["Close"], window=20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close=df["Close"], window=50).ema_indicator().iloc[-1]
    price = df["Close"].iloc[-1]

    has_ema200 = len(df) >= 200
    ema200 = EMAIndicator(close=df["Close"], window=200).ema_indicator().iloc[-1] if has_ema200 else None

    if price > ema20 > ema50:
        signal, note = 1, "ترتيب EMA صاعد"
    elif price < ema20 < ema50:
        signal, note = -1, "ترتيب EMA هابط"
    elif price > ema50:
        signal, note = 1, "السعر فوق EMA50"
    elif price < ema50:
        signal, note = -1, "السعر تحت EMA50"
    else:
        signal, note = 0, "EMA محايد"

    return IndicatorResult("EMA Trend", "المتوسطات الأسية", round(float(ema20), 2), signal, _signal_text(signal), note)


def calc_adx(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """ADX — trend strength."""
    adx_ind = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    adx = adx_ind.adx().iloc[-1]
    plus_di = adx_ind.adx_pos().iloc[-1]
    minus_di = adx_ind.adx_neg().iloc[-1]

    if adx > 25:
        if plus_di > minus_di:
            signal, note = 1, f"اتجاه صاعد قوي (ADX={adx:.1f})"
        else:
            signal, note = -1, f"اتجاه هابط قوي (ADX={adx:.1f})"
    elif adx > 20:
        if plus_di > minus_di:
            signal, note = 1, f"اتجاه صاعد متوسط (ADX={adx:.1f})"
        elif minus_di > plus_di:
            signal, note = -1, f"اتجاه هابط متوسط (ADX={adx:.1f})"
        else:
            signal, note = 0, f"اتجاه ضعيف (ADX={adx:.1f})"
    else:
        signal, note = 0, f"سوق متذبذب (ADX={adx:.1f})"

    return IndicatorResult("ADX", "مؤشر الاتجاه", round(float(adx), 2), signal, _signal_text(signal), note)


def calc_obv(df: pd.DataFrame) -> IndicatorResult:
    """On-Balance Volume — accumulation vs distribution."""
    obv_ind = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    obv = obv_ind.on_balance_volume().iloc[-1]
    obv_series = obv_ind.on_balance_volume().tail(10)
    obv_slope = (obv_series.iloc[-1] - obv_series.iloc[0]) / len(obv_series)

    avg_vol = df["Volume"].tail(20).mean()
    obv_normalized = obv_slope / avg_vol if avg_vol > 0 else 0

    if obv_normalized > 0.5:
        signal, note = 1, "تراكم قوي — تدفق سيولة شرائية"
    elif obv_normalized > 0.1:
        signal, note = 1, "تراكم إيجابي"
    elif obv_normalized < -0.5:
        signal, note = -1, "توزيع قوي — خروج سيولة"
    elif obv_normalized < -0.1:
        signal, note = -1, "توزيع سلبي"
    else:
        signal, note = 0, "تدفق محايد"

    return IndicatorResult("OBV", "مؤشر التوازن الحجمي", round(float(obv)), signal, _signal_text(signal), note)


def calc_williams_r(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """Williams %R — oversold < -80, overbought > -20."""
    wr_ind = WilliamsRIndicator(high=df["High"], low=df["Low"], close=df["Close"], lbp=period)
    wr = wr_ind.williams_r().iloc[-1]

    if wr < -80:
        signal, note = 1, "تشبع بيعي"
    elif wr > -20:
        signal, note = -1, "تشبع شرائي"
    elif wr < -50:
        signal, note = 0, "منطقة دنيا"
    else:
        signal, note = 0, "منطقة عليا"

    return IndicatorResult("Williams %R", "ويليامز %R", round(float(wr), 2), signal, _signal_text(signal), note)


# ─── Advanced Indicators ─────────────────────────────────────────────────────

def calc_vwap(df: pd.DataFrame, window: int = 20) -> IndicatorResult:
    """VWAP — Volume Weighted Average Price over rolling window."""
    recent = df.tail(window)
    typical_price = (recent["High"] + recent["Low"] + recent["Close"]) / 3
    vwap = (typical_price * recent["Volume"]).sum() / recent["Volume"].sum()
    price = df["Close"].iloc[-1]

    if price > vwap:
        signal, note = 1, f"السعر فوق VWAP ({vwap:.2f}) — زخم شرائي"
    elif price < vwap:
        signal, note = -1, f"السعر تحت VWAP ({vwap:.2f}) — ضغط بيعي"
    else:
        signal, note = 0, f"السعر عند VWAP ({vwap:.2f})"

    return IndicatorResult("VWAP", "متوسط السعر المرجح بالحجم", round(float(vwap), 2), signal, _signal_text(signal), note)


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> IndicatorResult:
    """SuperTrend — ATR-based trend indicator."""
    atr_ind = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    atr = atr_ind.average_true_range()

    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Final upper/lower bands (carry forward logic)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    trend = pd.Series(index=df.index, dtype=float)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        # Final upper band
        if upper_band.iloc[i] == upper_band.iloc[i-1] or pd.isna(final_upper.iloc[i-1]):
            pass
        elif df["Close"].iloc[i-1] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]

        # Final lower band
        if lower_band.iloc[i] == lower_band.iloc[i-1] or pd.isna(final_lower.iloc[i-1]):
            pass
        elif df["Close"].iloc[i-1] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]

    # Determine trend
    prev_trend = 1  # Start bullish
    for i in range(len(df)):
        if pd.isna(final_upper.iloc[i]) or pd.isna(final_lower.iloc[i]):
            continue
        if df["Close"].iloc[i] > final_upper.iloc[i]:
            trend.iloc[i] = 1
        elif df["Close"].iloc[i] < final_lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = prev_trend
        prev_trend = trend.iloc[i]

    current_trend = trend.iloc[-1] if not pd.isna(trend.iloc[-1]) else 0
    current_atr_pct = (atr.iloc[-1] / df["Close"].iloc[-1] * 100) if df["Close"].iloc[-1] > 0 else 0

    if current_trend == 1:
        signal, note = 1, f"SuperTrend صاعد (ATR={current_atr_pct:.1f}%)"
    elif current_trend == -1:
        signal, note = -1, f"SuperTrend هابط (ATR={current_atr_pct:.1f}%)"
    else:
        signal, note = 0, "SuperTrend محايد"

    return IndicatorResult("SuperTrend", "سوبر ترند", round(float(current_atr_pct), 2), signal, _signal_text(signal), note)


def calc_support_resistance(df: pd.DataFrame, lookback: int = 60, pivot_strength: int = 3) -> SupportResistanceResult:
    """
    Find support and resistance levels using swing highs/lows.
    A swing high is a bar whose High is higher than `pivot_strength` bars on each side.
    """
    recent = df.tail(lookback)
    price = df["Close"].iloc[-1]

    highs = []
    lows = []

    for i in range(pivot_strength, len(recent) - pivot_strength):
        # Swing high
        is_high = True
        for j in range(1, pivot_strength + 1):
            if recent["High"].iloc[i] <= recent["High"].iloc[i - j] or recent["High"].iloc[i] <= recent["High"].iloc[i + j]:
                is_high = False
                break
        if is_high:
            highs.append(recent["High"].iloc[i])

        # Swing low
        is_low = True
        for j in range(1, pivot_strength + 1):
            if recent["Low"].iloc[i] >= recent["Low"].iloc[i - j] or recent["Low"].iloc[i] >= recent["Low"].iloc[i + j]:
                is_low = False
                break
        if is_low:
            lows.append(recent["Low"].iloc[i])

    # Nearest resistance above price
    resistance = min([h for h in highs if h > price], default=0.0)
    # Nearest support below price
    support = max([l for l in lows if l < price], default=0.0)

    dist_sup = ((price - support) / price * 100) if support > 0 else 0
    dist_res = ((resistance - price) / price * 100) if resistance > 0 else 0

    return SupportResistanceResult(
        support=round(float(support), 2),
        resistance=round(float(resistance), 2),
        distance_to_support_pct=round(float(dist_sup), 2),
        distance_to_resistance_pct=round(float(dist_res), 2),
    )


def calc_breakout(sr: SupportResistanceResult, price: float, volume: int, avg_volume: float) -> IndicatorResult:
    """Detect breakout above resistance or breakdown below support."""
    if sr.resistance > 0 and price > sr.resistance:
        vol_confirm = volume > avg_volume * 1.3 if avg_volume > 0 else False
        note = "اختراق مقاومة" + (" بتأكيد حجمي" if vol_confirm else " بدون تأكيد حجمي")
        signal = 1
    elif sr.support > 0 and price < sr.support:
        vol_confirm = volume > avg_volume * 1.3 if avg_volume > 0 else False
        note = "كسر دعم" + (" بتأكيد حجمي" if vol_confirm else " بدون تأكيد حجمي")
        signal = -1
    else:
        signal = 0
        note = "السعر داخل النطاق"

    return IndicatorResult("Breakout", "الاختراق", round(float(price), 2), signal, _signal_text(signal), note)


def calc_risk_reward(sr: SupportResistanceResult, price: float) -> IndicatorResult:
    """Risk/Reward ratio based on support (risk) and resistance (reward)."""
    if sr.support <= 0 or sr.resistance <= 0 or price <= 0:
        return IndicatorResult("Risk/Reward", "المخاطرة/العائد", 0.0, 0, "محايد", "لا تتوفر مستويات واضحة")

    risk = price - sr.support
    reward = sr.resistance - price

    if risk <= 0:
        # Price below support — unusual, high risk
        return IndicatorResult("Risk/Reward", "المخاطرة/العائد", 0.0, -1, "هابط", "السعر تحت الدعم")

    rr_ratio = reward / risk

    if rr_ratio >= 2.0:
        signal, note = 1, f"نسبة ممتازة ({rr_ratio:.1f}:1)"
    elif rr_ratio >= 1.0:
        signal, note = 0, f"نسبة مقبولة ({rr_ratio:.1f}:1)"
    else:
        signal, note = -1, f"نسبة ضعيفة ({rr_ratio:.1f}:1)"

    return IndicatorResult("Risk/Reward", "المخاطرة/العائد", round(float(rr_ratio), 2), signal, _signal_text(signal), note)


def calc_volume_ratio(df: pd.DataFrame) -> IndicatorResult:
    """Volume ratio: current volume vs 20-day average."""
    current_vol = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].tail(20).mean()

    if avg_vol <= 0:
        return IndicatorResult("Volume Ratio", "نسبة الحجم", 1.0, 0, "محايد", "لا تتوفر بيانات حجم")

    ratio = current_vol / avg_vol

    if ratio > 2.0:
        signal, note = 1, f"حجم استثنائي ({ratio:.1f}x)"
    elif ratio > 1.5:
        signal, note = 1, f"حجم مرتفع ({ratio:.1f}x)"
    elif ratio < 0.5:
        signal, note = -1, f"حجم منخفض ({ratio:.1f}x)"
    else:
        signal, note = 0, f"حجم طبيعي ({ratio:.1f}x)"

    return IndicatorResult("Volume Ratio", "نسبة الحجم", round(float(ratio), 2), signal, _signal_text(signal), note)


def calc_volume_profile(df: pd.DataFrame, bins: int = 30, lookback: int = 60) -> VolumeProfileResult:
    """Simplified Volume Profile — POC and Value Area."""
    recent = df.tail(lookback).copy()
    price_min = recent["Low"].min()
    price_max = recent["High"].max()
    current_price = df["Close"].iloc[-1]

    if price_max == price_min:
        return VolumeProfileResult(
            poc=float(current_price), value_area_high=float(current_price),
            value_area_low=float(current_price), current_price_position="داخل منطقة القيمة",
            signal=0, signal_text="محايد",
        )

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    volume_per_bin = np.zeros(bins)

    for _, row in recent.iterrows():
        low, high, vol = row["Low"], row["High"], row["Volume"]
        price_range = high - low
        if price_range <= 0:
            idx = max(0, min(bins - 1, int(np.searchsorted(bin_edges, row["Close"]) - 1)))
            volume_per_bin[idx] += vol
        else:
            for i in range(bins):
                overlap = min(high, bin_edges[i + 1]) - max(low, bin_edges[i])
                if overlap > 0:
                    volume_per_bin[i] += vol * (overlap / price_range)

    poc_idx = int(np.argmax(volume_per_bin))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    total_vol = volume_per_bin.sum()

    if total_vol == 0:
        return VolumeProfileResult(
            poc=round(float(poc_price), 2),
            value_area_high=round(float(bin_edges[-1]), 2),
            value_area_low=round(float(bin_edges[0]), 2),
            current_price_position="داخل منطقة القيمة",
            signal=0, signal_text="محايد",
        )

    target_vol = total_vol * 0.70
    va_indices = [poc_idx]
    va_vol = volume_per_bin[poc_idx]
    low_idx, high_idx = poc_idx - 1, poc_idx + 1

    while va_vol < target_vol and (low_idx >= 0 or high_idx < bins):
        low_vol = volume_per_bin[low_idx] if low_idx >= 0 else -1
        high_vol = volume_per_bin[high_idx] if high_idx < bins else -1

        if low_vol >= high_vol and low_idx >= 0:
            va_indices.append(low_idx)
            va_vol += low_vol
            low_idx -= 1
        elif high_vol > low_vol and high_idx < bins:
            va_indices.append(high_idx)
            va_vol += high_vol
            high_idx += 1
        elif low_idx >= 0:
            va_indices.append(low_idx)
            va_vol += low_vol
            low_idx -= 1
        elif high_idx < bins:
            va_indices.append(high_idx)
            va_vol += high_vol
            high_idx += 1
        else:
            break

    va_low = float(bin_edges[min(va_indices)])
    va_high = float(bin_edges[max(va_indices) + 1])

    if current_price > va_high:
        position, signal, signal_text = "فوق منطقة القيمة", 1, "صاعد"
    elif current_price < va_low:
        position, signal, signal_text = "تحت منطقة القيمة", -1, "هابط"
    else:
        position, signal, signal_text = "داخل منطقة القيمة", 0, "محايد"

    return VolumeProfileResult(
        poc=round(float(poc_price), 2),
        value_area_high=round(va_high, 2),
        value_area_low=round(va_low, 2),
        current_price_position=position,
        signal=signal, signal_text=signal_text,
    )


def calc_atr(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """ATR — volatility (informational, not directional)."""
    atr_ind = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    atr = atr_ind.average_true_range().iloc[-1]
    price = df["Close"].iloc[-1]
    atr_pct = (atr / price * 100) if price > 0 else 0

    if atr_pct > 4:
        note = "تذبذب مرتفع"
    elif atr_pct > 2:
        note = "تذبذب متوسط"
    else:
        note = "تذبذب منخفض"

    return IndicatorResult("ATR", "متوسط المدى الحقيقي", round(float(atr_pct), 2), 0, "معلومة", note)


# ─── Composite Analysis ───────────────────────────────────────────────────────

def analyze_stock(
    df: pd.DataFrame,
    ticker: str,
    name: str = "",
    name_ar: str = "",
) -> StockAnalysis:
    """
    Run full technical analysis on a stock's OHLCV data.
    Returns a StockAnalysis with all indicators and extended fields.
    """
    if df is None or len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: {len(df) if df is not None else 0} rows")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing} in data for {ticker}")

    df = df.dropna(subset=required)
    if len(df) < 50:
        raise ValueError(f"Insufficient non-NaN data for {ticker}: {len(df)} rows after dropna")

    df = df.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})

    current_price = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    daily_change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
    volume = int(df["Volume"].iloc[-1])

    analysis = StockAnalysis(
        ticker=ticker,
        name=name or ticker,
        name_ar=name_ar or ticker,
        current_price=round(current_price, 2),
        daily_change_pct=round(daily_change_pct, 2),
        volume=volume,
    )

    # ── Calculate all indicators ──
    indicator_fns = [
        ("RSI", calc_rsi),
        ("Stochastic", calc_stochastic),
        ("Stochastic RSI", calc_stochastic_rsi),
        ("MACD", calc_macd),
        ("Bollinger", calc_bollinger),
        ("SMA Trend", calc_sma_trend),
        ("EMA Trend", calc_ema_trend),
        ("ADX", calc_adx),
        ("OBV", calc_obv),
        ("Williams %R", calc_williams_r),
        ("VWAP", calc_vwap),
        ("SuperTrend", calc_supertrend),
        ("Volume Ratio", calc_volume_ratio),
        ("ATR", calc_atr),
    ]

    for ind_name, fn in indicator_fns:
        try:
            analysis.indicators.append(fn(df))
        except Exception as e:
            logger.warning(f"{ind_name} failed for {ticker}: {e}")

    # Volume Profile
    try:
        analysis.volume_profile = calc_volume_profile(df)
    except Exception as e:
        logger.warning(f"Volume Profile failed for {ticker}: {e}")

    # Support/Resistance
    try:
        sr = calc_support_resistance(df)
        analysis.support_resistance = sr
        analysis.support = sr.support
        analysis.resistance = sr.resistance
    except Exception as e:
        logger.warning(f"Support/Resistance failed for {ticker}: {e}")
        sr = SupportResistanceResult(0, 0, 0, 0)

    # Breakout
    try:
        avg_vol = df["Volume"].tail(20).mean()
        analysis.indicators.append(calc_breakout(sr, current_price, volume, avg_vol))
    except Exception as e:
        logger.warning(f"Breakout detection failed for {ticker}: {e}")

    # Risk/Reward
    try:
        rr = calc_risk_reward(sr, current_price)
        analysis.indicators.append(rr)
        analysis.risk_reward_ratio = rr.value
    except Exception as e:
        logger.warning(f"Risk/Reward failed for {ticker}: {e}")

    # VWAP value
    vwap_ind = _get_ind(analysis, "VWAP")
    if vwap_ind:
        analysis.vwap = vwap_ind.value

    # SuperTrend signal
    st_ind = _get_ind(analysis, "SuperTrend")
    if st_ind:
        analysis.supertrend_signal = st_ind.signal_text

    # Breakout type
    breakout_ind = _get_ind(analysis, "Breakout")
    if breakout_ind:
        if breakout_ind.signal == 1:
            analysis.breakout_type = "bullish_breakout"
        elif breakout_ind.signal == -1:
            analysis.breakout_type = "bearish_breakout"

    # ── Composite score (from directional indicators only) ──
    directional = [i for i in analysis.indicators if i.name != "ATR" and i.name != "Risk/Reward"]
    scores = [i.signal for i in directional]

    if analysis.volume_profile:
        scores.append(analysis.volume_profile.signal)

    composite = sum(scores)
    analysis.composite_score = composite

    max_possible = len(scores)
    if max_possible > 0:
        analysis.signal_score_pct = round((composite / max_possible + 1) / 2 * 100, 1)
    else:
        analysis.signal_score_pct = 50.0

    if composite >= 4:
        analysis.signal_label = "شراء قوي 🟢🟢"
    elif composite >= 2:
        analysis.signal_label = "شراء 🟢"
    elif composite <= -4:
        analysis.signal_label = "بيع قوي 🔴🔴"
    elif composite <= -2:
        analysis.signal_label = "بيع 🔴"
    else:
        analysis.signal_label = "محايد 🟡"

    # Bullish/bearish reasons
    for ind in analysis.indicators:
        if ind.signal > 0 and ind.note:
            analysis.bullish_reasons.append(f"{ind.name_ar}: {ind.note}")
        elif ind.signal < 0 and ind.note:
            analysis.bearish_reasons.append(f"{ind.name_ar}: {ind.note}")

    if analysis.volume_profile:
        vp = analysis.volume_profile
        if vp.signal > 0:
            analysis.bullish_reasons.append(f"فوليوم بروفايل: {vp.current_price_position}")
        elif vp.signal < 0:
            analysis.bearish_reasons.append(f"فوليوم بروفايل: {vp.current_price_position}")

    return analysis


def _get_ind(analysis: StockAnalysis, name: str) -> IndicatorResult | None:
    """Find an indicator by name in the analysis."""
    for ind in analysis.indicators:
        if ind.name == name:
            return ind
    return None
