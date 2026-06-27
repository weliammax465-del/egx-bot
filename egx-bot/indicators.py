"""
indicators.py
-------------
Professional technical indicators for EGX stock analysis.

Indicators calculated:
  - RSI (14)
  - Stochastic Oscillator (14, 3, 3)
  - MACD (12, 26, 9)
  - Bollinger Bands (20, 2)
  - SMA (20, 50, 200)
  - EMA (12, 26)
  - ADX (14) — trend strength
  - OBV (On-Balance Volume)
  - Volume Profile (POC, Value Area, High Volume Nodes)
  - ATR (14) — volatility
  - Williams %R (14)

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
    note: str = ""       # Additional context (e.g., "تشبع بيعي")


@dataclass
class VolumeProfileResult:
    """Simplified Volume Profile analysis."""
    poc: float                  # Point of Control — price with highest volume
    value_area_high: float      # 70% value area upper bound
    value_area_low: float       # 70% value area lower bound
    current_price_position: str # "فوق منطقة القيمة", "داخل منطقة القيمة", "تحت منطقة القيمة"
    signal: int
    signal_text: str


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
    composite_score: int = 0
    signal_label: str = "محايد"       # "شراء قوي", "شراء", "محايد", "بيع", "بيع قوي"
    signal_score_pct: float = 0.0     # 0-100 confidence
    bullish_reasons: list[str] = field(default_factory=list)
    bearish_reasons: list[str] = field(default_factory=list)

    @property
    def is_bullish(self) -> bool:
        return self.composite_score >= 2

    @property
    def is_strong_bullish(self) -> bool:
        return self.composite_score >= 4


# ─── Indicator Calculations ───────────────────────────────────────────────────

def _signal_text(score: int) -> str:
    if score > 0:
        return "صاعد"
    elif score < 0:
        return "هابط"
    return "محايد"


def calc_rsi(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """RSI — oversold < 30, overbought > 70."""
    rsi_ind = RSIIndicator(close=df["Close"], window=period)
    rsi = rsi_ind.rsi().iloc[-1]

    if rsi < 30:
        signal, note = 1, "تشبع بيعي — احتمال ارتداد"
    elif rsi > 70:
        signal, note = -1, "تشبع شرائي — احتمال تصحيح"
    elif rsi < 45:
        signal, note = 1, "قوة شرائية ضعيفة"
    elif rsi > 55:
        signal, note = -1, "قوة شرائية قوية"
    else:
        signal, note = 0, "منطقة محايدة"

    return IndicatorResult(
        name="RSI", name_ar="مؤشر القوة النسبية",
        value=round(rsi, 2), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult:
    """Stochastic Oscillator — %K and %D crossover in oversold/overbought zones."""
    stoch = StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"],
        window=k_period, smooth_window=d_period,
    )
    k_val = stoch.stoch().iloc[-1]
    d_val = stoch.stoch_signal().iloc[-1]

    # Check crossover (current vs previous)
    k_prev = stoch.stoch().iloc[-2] if len(stoch.stoch()) > 1 else k_val
    d_prev = stoch.stoch_signal().iloc[-2] if len(stoch.stoch_signal()) > 1 else d_val

    bullish_cross = k_prev <= d_prev and k_val > d_val
    bearish_cross = k_prev >= d_prev and k_val < d_val

    if k_val < 20 and bullish_cross:
        signal, note = 1, f"تقاطع صاعد في منطقة التشبع البيعي (%K={k_val:.1f})"
    elif k_val > 80 and bearish_cross:
        signal, note = -1, f"تقاطع هابط في منطقة التشبع الشرائي (%K={k_val:.1f})"
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

    return IndicatorResult(
        name="Stochastic", name_ar="مؤشر الاستوكاستك",
        value=round(k_val, 2), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_macd(df: pd.DataFrame) -> IndicatorResult:
    """MACD — trend momentum via 12/26 EMA crossover."""
    macd_ind = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd().iloc[-1]
    signal_line = macd_ind.macd_signal().iloc[-1]
    histogram = macd_ind.macd_diff().iloc[-1]

    # Check crossover
    macd_prev = macd_ind.macd().iloc[-2] if len(macd_ind.macd()) > 1 else macd_line
    signal_prev = macd_ind.macd_signal().iloc[-2] if len(macd_ind.macd_signal()) > 1 else signal_line

    bullish_cross = macd_prev <= signal_prev and macd_line > signal_line
    bearish_cross = macd_prev >= signal_prev and macd_line < signal_line

    if bullish_cross and histogram > 0:
        signal, note = 1, "تقاطع صاعد — زخم إيجابي قوي"
    elif bearish_cross and histogram < 0:
        signal, note = -1, "تقاطع هابط — زخم سلبي قوي"
    elif macd_line > signal_line:
        signal, note = 1, "MACD فوق خط الإشارة — زخم إيجابي"
    elif macd_line < signal_line:
        signal, note = -1, "MACD تحت خط الإشارة — زخم سلبي"
    else:
        signal, note = 0, "زخم محايد"

    return IndicatorResult(
        name="MACD", name_ar="ماكد",
        value=round(macd_line, 4), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_bollinger(df: pd.DataFrame, window: int = 20, dev: int = 2) -> IndicatorResult:
    """Bollinger Bands — price relative to bands."""
    bb = BollingerBands(close=df["Close"], window=window, window_dev=dev)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    mid = bb.bollinger_mavg().iloc[-1]
    price = df["Close"].iloc[-1]

    # Position within bands (0 = lower band, 1 = upper band)
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

    return IndicatorResult(
        name="Bollinger", name_ar="بولينجر باند",
        value=round(position * 100, 1), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_sma_trend(df: pd.DataFrame) -> IndicatorResult:
    """SMA 20/50/200 trend analysis — golden cross / death cross."""
    sma20 = SMAIndicator(close=df["Close"], window=20).sma_indicator().iloc[-1]
    sma50 = SMAIndicator(close=df["Close"], window=50).sma_indicator().iloc[-1]

    price = df["Close"].iloc[-1]

    # SMA 200 only if enough data
    if len(df) >= 200:
        sma200 = SMAIndicator(close=df["Close"], window=200).sma_indicator().iloc[-1]
        has_sma200 = True
    else:
        sma200 = None
        has_sma200 = False

    # Determine trend
    bullish_stack = price > sma20 > sma50
    bearish_stack = price < sma20 < sma50

    if has_sma200 and sma50 > sma200:
        golden_cross = True
    else:
        golden_cross = False

    if has_sma200 and sma50 < sma200:
        death_cross = True
    else:
        death_cross = False

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

    return IndicatorResult(
        name="SMA Trend", name_ar="المتوسطات المتحركة",
        value=round(sma20, 2), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_adx(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """ADX — trend strength (not direction). >25 = strong trend."""
    adx_ind = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    adx = adx_ind.adx().iloc[-1]
    plus_di = adx_ind.adx_pos().iloc[-1]
    minus_di = adx_ind.adx_neg().iloc[-1]

    if adx > 25:
        if plus_di > minus_di:
            signal, note = 1, f"اتجاه صاعد قوي (ADX={adx:.1f}, +DI > -DI)"
        else:
            signal, note = -1, f"اتجاه هابط قوي (ADX={adx:.1f}, -DI > +DI)"
    elif adx > 20:
        if plus_di > minus_di:
            signal, note = 1, f"اتجاه صاعد متوسط القوة (ADX={adx:.1f})"
        elif minus_di > plus_di:
            signal, note = -1, f"اتجاه هابط متوسط القوة (ADX={adx:.1f})"
        else:
            signal, note = 0, f"اتجاه ضعيف (ADX={adx:.1f})"
    else:
        signal, note = 0, f"سوق متذبذب بدون اتجاه واضح (ADX={adx:.1f})"

    return IndicatorResult(
        name="ADX", name_ar="مؤشر الاتجاه الاتجاهي",
        value=round(adx, 2), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_obv(df: pd.DataFrame) -> IndicatorResult:
    """On-Balance Volume — accumulation vs distribution."""
    obv_ind = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    obv = obv_ind.on_balance_volume().iloc[-1]

    # Check OBV trend (last 10 days)
    obv_series = obv_ind.on_balance_volume().tail(10)
    obv_slope = (obv_series.iloc[-1] - obv_series.iloc[0]) / len(obv_series)

    # Normalize by average volume
    avg_vol = df["Volume"].tail(20).mean()
    if avg_vol > 0:
        obv_normalized = obv_slope / avg_vol
    else:
        obv_normalized = 0

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

    return IndicatorResult(
        name="OBV", name_ar="مؤشر التوازن الحجمي",
        value=round(obv), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


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

    return IndicatorResult(
        name="Williams %R", name_ar="ويليامز %R",
        value=round(wr, 2), signal=signal,
        signal_text=_signal_text(signal), note=note,
    )


def calc_volume_profile(df: pd.DataFrame, bins: int = 30, lookback: int = 60) -> VolumeProfileResult:
    """
    Simplified Volume Profile:
    - Divide price range into bins
    - Sum volume at each price level
    - Find POC (Point of Control) and Value Area (70%)
    """
    recent = df.tail(lookback).copy()
    price_min = recent["Low"].min()
    price_max = recent["High"].max()
    current_price = df["Close"].iloc[-1]

    if price_max == price_min:
        return VolumeProfileResult(
            poc=current_price, value_area_high=current_price,
            value_area_low=current_price, current_price_position="داخل منطقة القيمة",
            signal=0, signal_text="محايد",
        )

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    volume_per_bin = np.zeros(bins)

    for _, row in recent.iterrows():
        # Distribute row's volume across bins it touches
        low = row["Low"]
        high = row["High"]
        vol = row["Volume"]
        price_range = high - low
        if price_range <= 0:
            # Assign to single bin
            idx = np.searchsorted(bin_edges, current_price) - 1
            idx = max(0, min(bins - 1, idx))
            volume_per_bin[idx] += vol
        else:
            for i in range(bins):
                overlap = min(high, bin_edges[i + 1]) - max(low, bin_edges[i])
                if overlap > 0:
                    volume_per_bin[i] += vol * (overlap / price_range)

    # POC — bin with highest volume
    poc_idx = np.argmax(volume_per_bin)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    # Value Area — 70% of volume around POC
    total_vol = volume_per_bin.sum()
    target_vol = total_vol * 0.70

    # Expand from POC outward
    va_indices = [poc_idx]
    va_vol = volume_per_bin[poc_idx]
    low_idx = poc_idx - 1
    high_idx = poc_idx + 1

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

    va_low = bin_edges[min(va_indices)]
    va_high = bin_edges[max(va_indices) + 1]

    # Position relative to value area
    if current_price > va_high:
        position = "فوق منطقة القيمة"
        signal = 1  # Price breaking above — bullish
        signal_text = "صاعد"
    elif current_price < va_low:
        position = "تحت منطقة القيمة"
        signal = -1  # Price breaking below — bearish
        signal_text = "هابط"
    else:
        position = "داخل منطقة القيمة"
        signal = 0
        signal_text = "محايد"

    return VolumeProfileResult(
        poc=round(poc_price, 2),
        value_area_high=round(va_high, 2),
        value_area_low=round(va_low, 2),
        current_price_position=position,
        signal=signal,
        signal_text=signal_text,
    )


def calc_atr(df: pd.DataFrame, period: int = 14) -> IndicatorResult:
    """ATR — volatility measure (informational, not directional)."""
    atr_ind = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=period)
    atr = atr_ind.average_true_range().iloc[-1]
    price = df["Close"].iloc[-1]

    atr_pct = (atr / price) * 100 if price > 0 else 0

    if atr_pct > 4:
        note = "تذبذب مرتفع"
    elif atr_pct > 2:
        note = "تذبذب متوسط"
    else:
        note = "تذبذب منخفض"

    # ATR is informational — no buy/sell signal
    return IndicatorResult(
        name="ATR", name_ar="متوسط المدى الحقيقي",
        value=round(atr_pct, 2), signal=0,
        signal_text="معلومة", note=note,
    )


# ─── Composite Analysis ───────────────────────────────────────────────────────

def analyze_stock(
    df: pd.DataFrame,
    ticker: str,
    name: str = "",
    name_ar: str = "",
) -> StockAnalysis:
    """
    Run full technical analysis on a stock's OHLCV data.
    Returns a StockAnalysis with all indicators and a composite score.
    """
    if df is None or len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: {len(df) if df is not None else 0} rows")

    # Flatten multi-index columns if yfinance returned them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Ensure correct dtypes
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in data for {ticker}")

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
    try:
        analysis.indicators.append(calc_rsi(df))
    except Exception as e:
        logger.warning(f"RSI failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_stochastic(df))
    except Exception as e:
        logger.warning(f"Stochastic failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_macd(df))
    except Exception as e:
        logger.warning(f"MACD failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_bollinger(df))
    except Exception as e:
        logger.warning(f"Bollinger failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_sma_trend(df))
    except Exception as e:
        logger.warning(f"SMA trend failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_adx(df))
    except Exception as e:
        logger.warning(f"ADX failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_obv(df))
    except Exception as e:
        logger.warning(f"OBV failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_williams_r(df))
    except Exception as e:
        logger.warning(f"Williams %R failed for {ticker}: {e}")

    try:
        analysis.volume_profile = calc_volume_profile(df)
    except Exception as e:
        logger.warning(f"Volume Profile failed for {ticker}: {e}")

    try:
        analysis.indicators.append(calc_atr(df))
    except Exception as e:
        logger.warning(f"ATR failed for {ticker}: {e}")

    # ── Composite score ──
    # Only count directional indicators (skip ATR which is informational)
    directional_indicators = [i for i in analysis.indicators if i.signal != 0 or i.name != "ATR"]
    scores = [i.signal for i in directional_indicators]

    # Add volume profile signal
    if analysis.volume_profile:
        scores.append(analysis.volume_profile.signal)

    composite = sum(scores)
    analysis.composite_score = composite

    # Signal label based on composite score
    max_possible = len(scores)  # Maximum possible score
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

    # Collect bullish/bearish reasons
    for ind in analysis.indicators:
        if ind.signal > 0 and ind.note:
            analysis.bullish_reasons.append(f"{ind.name_ar}: {ind.note}")
        elif ind.signal < 0 and ind.note:
            analysis.bearish_reasons.append(f"{ind.name_ar}: {ind.note}")

    if analysis.volume_profile:
        vp = analysis.volume_profile
        if vp.signal > 0:
            analysis.bullish_reasons.append(f"فوليوم بروفايل: السعر {vp.current_price_position}")
        elif vp.signal < 0:
            analysis.bearish_reasons.append(f"فوليوم بروفايل: السعر {vp.current_price_position}")

    return analysis
