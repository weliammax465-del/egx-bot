"""
EMA50 Breakout Strategy — Pine Script translation for EGX Bot
===============================================================
Strategy flow:
1. Stock trades below EMA50 for N candles (precondition)
2. Breakout candle: close crosses above EMA50 -> record breakout_high
3. waiting_confirmation = True
4. Check ALL 8 conditions on latest candle:
   - cond1: candles_below_count >= 2 (was below EMA before breakout)
   - cond2: close > EMA50 (price above EMA)
   - cond3: close > breakout_high (broke above breakout candle's high)
   - cond4: RSI > threshold (momentum)
   - cond5: +DI > -DI (bullish direction)
   - cond6: ADX > threshold (trend strength)
   - cond7: volume > volume_avg (volume confirmation)
   - cond8: price hasn't returned below EMA (breakout held)
5. ALL met -> BUY signal; price falls below EMA -> cancel
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Strategy Parameters ─────────────────────────────────────────────────────

EMA_PERIOD = 50
RSI_THRESHOLD = 50.0
ADX_THRESHOLD = 20.0
VOLUME_AVG_PERIOD = 20
MIN_CANDLES_BELOW = 2
MIN_BARS_REQUIRED = 60
EMA_TOLERANCE = 0.005  # 0.5% tolerance for "below EMA" check (avoids noise)


# ─── Data Class ──────────────────────────────────────────────────────────────

@dataclass
class BreakoutSignal:
    ticker: str
    signal: str = "NO_SIGNAL"
    signal_ar: str = "لا توجد إشارة"
    cond1: bool = False
    cond2: bool = False
    cond3: bool = False
    cond4: bool = False
    cond5: bool = False
    cond6: bool = False
    cond7: bool = False
    cond8: bool = False
    ema_value: float = 0.0
    close: float = 0.0
    breakout_high: float = 0.0
    breakout_date: str = ""
    candles_below_count: int = 0
    rsi: float = 0.0
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0
    volume: float = 0.0
    volume_avg: float = 0.0
    waiting_confirmation: bool = False
    stop_loss: float = 0.0
    target: float = 0.0
    atr: float = 0.0
    data_date: str = ""
    error: str = ""

    @property
    def conditions_met(self) -> int:
        return sum([self.cond1, self.cond2, self.cond3, self.cond4,
                    self.cond5, self.cond6, self.cond7, self.cond8])

    @property
    def is_buy(self) -> bool:
        return self.signal == "BUY"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "signal": self.signal,
            "conditions_met": self.conditions_met,
            "ema": round(self.ema_value, 2), "close": round(self.close, 2),
            "breakout_high": round(self.breakout_high, 2),
            "rsi": round(self.rsi, 1), "adx": round(self.adx, 1),
            "di_plus": round(self.di_plus, 1), "di_minus": round(self.di_minus, 1),
            "volume": int(self.volume), "volume_avg": int(self.volume_avg),
            "stop_loss": round(self.stop_loss, 2), "target": round(self.target, 2),
        }


# ─── Technical Indicators ────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


# ─── Core Strategy Engine ────────────────────────────────────────────────────

def evaluate_breakout(df: pd.DataFrame, ticker: str) -> BreakoutSignal:
    """Evaluate the EMA50 breakout strategy for a single stock."""
    result = BreakoutSignal(ticker=ticker)

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('open', 'high', 'low', 'close', 'volume'):
            col_map[c] = cl.capitalize()
    df = df.rename(columns=col_map)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.error = f"Missing columns: {missing}"
        return result

    if len(df) < MIN_BARS_REQUIRED:
        result.error = f"Insufficient data: {len(df)} bars (need {MIN_BARS_REQUIRED})"
        return result

    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
    if len(df) < MIN_BARS_REQUIRED:
        result.error = f"Insufficient valid data after dropna: {len(df)}"
        return result

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ema = _ema(close, EMA_PERIOD)
    rsi = _rsi(close, 14)
    adx_s, di_plus_s, di_minus_s = _adx(high, low, close, 14)
    atr_s = _atr(high, low, close, 14)
    vol_avg = volume.rolling(VOLUME_AVG_PERIOD).mean()

    # ── State Machine (skip warmup: first EMA_PERIOD bars) ────────────────────
    # Key fixes vs v1:
    # 1. Skip first EMA_PERIOD bars so EMA is stabilized
    # 2. Tolerance for "below EMA" check (0.1%) to avoid noise canceling breakouts
    # 3. Don't overwrite pre_breakout_count on false returns; keep the LATEST breakout
    # 4. Only update pre_breakout_count when a NEW breakout fires

    candles_below_count = 0
    breakout_high = 0.0
    breakout_date = ""
    breakout_idx = -1
    waiting = False
    returned_below = False
    pre_breakout_count = 0
    highest_since_breakout = 0.0

    start_bar = EMA_PERIOD  # skip warmup

    for i in range(len(df)):
        if i < start_bar:
            continue
        if pd.isna(ema.iloc[i]):
            continue

        c = float(close.iloc[i])
        h = float(high.iloc[i])
        e = float(ema.iloc[i])
        tolerance = e * EMA_TOLERANCE

        if c < e - tolerance:
            # Below EMA (with tolerance)
            if not waiting:
                candles_below_count += 1
            else:
                # Returned below EMA while waiting -> cancel
                returned_below = True
                waiting = False
                candles_below_count = 1  # start counting new below-period
        elif c > e + tolerance:
            # Above EMA (with tolerance)
            if not waiting and candles_below_count >= MIN_CANDLES_BELOW:
                # NEW breakout!
                breakout_high = h
                breakout_date = str(df.index[i])[:10] if not isinstance(df.index[i], (int, float)) else str(i)
                breakout_idx = i
                pre_breakout_count = candles_below_count
                waiting = True
                returned_below = False  # reset for new breakout cycle
                highest_since_breakout = h
                candles_below_count = 0
            elif waiting:
                if h > highest_since_breakout:
                    highest_since_breakout = h
        # else: within tolerance, treat as "at EMA" — don't change state

    # ── Evaluate Conditions (latest candle) ───────────────────────────────────
    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_adx = float(adx_s.iloc[-1])
    last_di_plus = float(di_plus_s.iloc[-1])
    last_di_minus = float(di_minus_s.iloc[-1])
    last_vol = float(volume.iloc[-1])
    last_vol_avg = float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0
    last_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    data_date = str(df.index[-1])[:10] if not isinstance(df.index[-1], (int, float)) else ""

    result.close = last_close
    result.ema_value = last_ema
    result.rsi = last_rsi
    result.adx = last_adx
    result.di_plus = last_di_plus
    result.di_minus = last_di_minus
    result.volume = last_vol
    result.volume_avg = last_vol_avg
    result.atr = last_atr
    result.data_date = data_date
    result.waiting_confirmation = waiting
    result.breakout_high = breakout_high
    result.breakout_date = breakout_date
    result.candles_below_count = pre_breakout_count if breakout_idx >= 0 else candles_below_count

    # ── 8 Conditions ──────────────────────────────────────────────────────────
    result.cond1 = (pre_breakout_count >= MIN_CANDLES_BELOW) if breakout_idx >= 0 else (candles_below_count >= MIN_CANDLES_BELOW)
    result.cond2 = last_close > last_ema
    result.cond3 = waiting and (last_close > breakout_high) if breakout_high > 0 else False
    result.cond4 = last_rsi > RSI_THRESHOLD
    result.cond5 = last_di_plus > last_di_minus
    result.cond6 = last_adx > ADX_THRESHOLD
    result.cond7 = last_vol > last_vol_avg if last_vol_avg > 0 else False
    result.cond8 = waiting and not returned_below

    # ── Final Signal ──────────────────────────────────────────────────────────
    all_met = (result.cond1 and result.cond2 and result.cond3 and result.cond4
               and result.cond5 and result.cond6 and result.cond7 and result.cond8)

    if all_met:
        result.signal = "BUY"
        result.signal_ar = "🟢 شراء"
        if last_atr > 0:
            result.stop_loss = last_close - (1.5 * last_atr)
            risk = last_close - result.stop_loss
            result.target = last_close + (2.0 * risk)
        elif breakout_high > 0:
            result.stop_loss = breakout_high * 0.97
            result.target = last_close + (last_close - result.stop_loss) * 2
    elif waiting and not returned_below:
        result.signal = "WAIT"
        result.signal_ar = "⏸ انتظار تأكيد"
    else:
        result.signal = "NO_SIGNAL"
        result.signal_ar = "لا توجد إشارة"

    return result


def scan_stocks_breakout(stock_list, download_func, progress_callback=None):
    """Scan all stocks using the EMA50 breakout strategy."""
    results = []
    total = len(stock_list)
    for i, stock_info in enumerate(stock_list, 1):
        ticker = stock_info["symbol"]
        if progress_callback and i % 20 == 0:
            progress_callback(i, total, len(results))
        try:
            df = download_func(ticker, n_bars=250, retries=2)
            if df is None or df.empty:
                logger.debug(f"  {ticker}: no data")
                continue
            signal = evaluate_breakout(df, ticker)
            if stock_info.get("price", 0) > 0:
                signal.close = stock_info["price"]
            results.append(signal)
        except Exception as e:
            logger.warning(f"  {ticker}: error — {str(e)[:80]}")
    return results


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_breakout_summary(signals):
    """Format breakout results for Telegram message."""
    buy_signals = [s for s in signals if s.is_buy]
    wait_signals = [s for s in signals if s.signal == "WAIT"]
    no_signal = [s for s in signals if s.signal == "NO_SIGNAL"]
    errors = [s for s in signals if s.error]

    lines = ["📊 *تقرير EGX — استراتيجية اختراق EMA50*", ""]
    lines.append(f"📈 النتائج: {len(buy_signals)} شراء | {len(wait_signals)} انتظار | {len(no_signal)} لا إشارة | {len(errors)} خطأ")
    lines.append("")

    if buy_signals:
        lines.append("🟢 *إشارات شراء مؤكدة:*")
        lines.append("")
        for i, s in enumerate(buy_signals, 1):
            lines.append(f"{i}. *{s.ticker}* — {s.close:.2f} EGP")
            lines.append(f"   🎯 اختراق: {s.breakout_high:.2f} ({s.breakout_date})")
            lines.append(f"   📊 RSI: {s.rsi:.0f} | ADX: {s.adx:.0f} | +DI: {s.di_plus:.0f} > -DI: {s.di_minus:.0f}")
            lines.append(f"   📦 حجم: {int(s.volume):,} (متوسط: {int(s.volume_avg):,})")
            if s.stop_loss > 0:
                lines.append(f"   🛑 وقف: {s.stop_loss:.2f} | 🎯 هدف: {s.target:.2f}")
            conds = []
            for label, val in [("أسفلEMA", s.cond1), ("فوقEMA", s.cond2), ("اختراق", s.cond3),
                               ("RSI", s.cond4), ("+DI", s.cond5), ("ADX", s.cond6),
                               ("حجم", s.cond7), ("ثبات", s.cond8)]:
                conds.append(f"{label}={'✅' if val else '❌'}")
            lines.append(f"   {' | '.join(conds)}")
            lines.append("")
    else:
        lines.append("⚪ لا توجد إشارات شراء مؤكدة اليوم.")
        lines.append("")

    if wait_signals:
        lines.append("⏸ *في انتظار التأكيد:*")
        lines.append("")
        for s in sorted(wait_signals, key=lambda x: x.conditions_met, reverse=True)[:10]:
            met = s.conditions_met
            lines.append(f"   • {s.ticker} — {s.close:.2f} | {met}/8 شروط | اختراق: {s.breakout_high:.2f}")
        if len(wait_signals) > 10:
            lines.append(f"   ... و {len(wait_signals) - 10} سهم آخر")
        lines.append("")

    lines += [
        "─────────────────────",
        "📊 الاستراتيجية: اختراق EMA50 + RSI + ADX + حجم",
        f"⚙️ EMA={EMA_PERIOD} | RSI>{RSI_THRESHOLD} | ADX>{ADX_THRESHOLD}",
        "⚠️ هذه ليست نصيحة استثمارية",
    ]
    return "\n".join(lines)


def format_stock_breakout_detail(signal):
    """Format a single stock's breakout analysis for /stock command."""
    lines = [
        f"📊 *{signal.ticker}*",
        "",
        f"💰 السعر: *{signal.close:.2f} EGP*",
        f"📈 EMA{EMA_PERIOD}: {signal.ema_value:.2f}",
        f"📅 البيانات: {signal.data_date}",
        "",
    ]
    if signal.error:
        lines.append(f"❌ خطأ: {signal.error}")
        return "\n".join(lines)

    lines.append(f"📋 الإشارة: {signal.signal_ar}")
    lines.append("")

    if signal.breakout_high > 0:
        lines.append(f"🎯 قمة الاختراق: {signal.breakout_high:.2f} ({signal.breakout_date})")
    lines.append(f"📊 شموع أسفل EMA قبل الاختراق: {signal.candles_below_count}")
    lines.append("")

    lines.append("🔬 *المؤشرات:*")
    lines.append(f"   RSI: {signal.rsi:.1f} (>{RSI_THRESHOLD}{' ✅' if signal.cond4 else ' ❌'})")
    lines.append(f"   ADX: {signal.adx:.1f} (>{ADX_THRESHOLD}{' ✅' if signal.cond6 else ' ❌'})")
    lines.append(f"   +DI: {signal.di_plus:.1f} | -DI: {signal.di_minus:.1f} ({'✅' if signal.cond5 else '❌'})")
    lines.append(f"   حجم: {int(signal.volume):,} (متوسط: {int(signal.volume_avg):,}) {'✅' if signal.cond7 else '❌'}")
    lines.append("")

    lines.append("📋 *الشروط (8/8 للشراء):*")
    conditions = [
        ("شموع أسفل EMA", signal.cond1, str(signal.candles_below_count)),
        ("السعر فوق EMA", signal.cond2, f"{signal.close:.2f} > {signal.ema_value:.2f}"),
        ("إغلاق > قمة الاختراق", signal.cond3, f"{signal.close:.2f} > {signal.breakout_high:.2f}" if signal.breakout_high else "لا يوجد"),
        (f"RSI > {RSI_THRESHOLD}", signal.cond4, f"{signal.rsi:.1f}"),
        ("+DI > -DI", signal.cond5, f"{signal.di_plus:.1f} > {signal.di_minus:.1f}"),
        (f"ADX > {ADX_THRESHOLD}", signal.cond6, f"{signal.adx:.1f}"),
        ("الحجم > المتوسط", signal.cond7, f"{int(signal.volume):,} > {int(signal.volume_avg):,}"),
        ("لم يعد أسفل EMA", signal.cond8, "نشط" if signal.cond8 else "ملغي"),
    ]
    for label, met, value in conditions:
        lines.append(f"   {'✅' if met else '❌'} {label}: {value}")
    lines.append("")

    if signal.stop_loss > 0:
        lines.append("⚖️ *إدارة المخاطر:*")
        lines.append(f"   🛑 وقف الخسارة: *{signal.stop_loss:.2f} EGP*")
        lines.append(f"   🎯 الهدف: *{signal.target:.2f} EGP*")
        lines.append("")

    lines += [
        "─────────────────────",
        f"⚙️ EMA{EMA_PERIOD} | RSI>{RSI_THRESHOLD} | ADX>{ADX_THRESHOLD}",
        "⚠️ للمعلومات فقط — ليست نصيحة استثمارية",
    ]
    return "\n".join(lines)


# ─── Pre-Breakout Scanner ────────────────────────────────────────────────────
# Finds stocks that are APPROACHING a breakout — still below EMA50 but showing
# accumulation signs (rising volume, tightening range, RSI climbing, ADX forming).
# This gives early warning BEFORE the breakout happens.

# Pre-breakout parameters
PROXIMITY_THRESHOLD = 5.0      # % below EMA50 to consider "approaching"
CONSOLIDATION_WINDOW = 20      # bars to measure consolidation tightness
VOLUME_TREND_WINDOW = 10       # bars to measure volume trend
RSI_RISING_WINDOW = 10         # bars to check if RSI is rising
ADX_FORMING_THRESHOLD = 15.0   # ADX rising above this = trend forming
MIN_PRE_BREAKOUT_SCORE = 45    # minimum score to include in report


@dataclass
class PreBreakoutSignal:
    """Stock approaching EMA50 breakout — hasn't broken out yet."""
    ticker: str
    signal: str = "NO_SETUP"
    signal_ar: str = "لا يوجد نمط"
    score: int = 0              # 0-100 proximity score (higher = closer to breakout)
    close: float = 0.0
    ema_value: float = 0.0
    distance_pct: float = 0.0   # how far below EMA50 (%)
    candles_below: int = 0      # how long below EMA50
    consolidation_pct: float = 0.0  # range tightness (lower = tighter = better)
    volume_trend: str = ""      # "rising" / "flat" / "falling"
    volume_ratio: float = 0.0   # recent vol / older vol
    rsi: float = 0.0
    rsi_trend: str = ""         # "rising" / "flat" / "falling"
    adx: float = 0.0
    adx_trend: str = ""         # "rising" / "flat" / "falling"
    di_plus: float = 0.0
    di_minus: float = 0.0
    higher_lows: bool = False   # making higher lows (bottoming)
    recent_high: float = 0.0    # recent high below EMA (resistance to watch)
    atr: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    data_date: str = ""
    error: str = ""

    @property
    def is_approaching(self) -> bool:
        return self.signal == "APPROACHING"

    @property
    def is_accumulating(self) -> bool:
        return self.signal == "ACCUMULATING"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "signal": self.signal, "score": self.score,
            "close": round(self.close, 2), "ema": round(self.ema_value, 2),
            "distance_pct": round(self.distance_pct, 2),
            "candles_below": self.candles_below,
            "volume_trend": self.volume_trend, "rsi": round(self.rsi, 1),
            "adx": round(self.adx, 1), "higher_lows": self.higher_lows,
            "recent_high": round(self.recent_high, 2),
        }


def evaluate_pre_breakout(df: pd.DataFrame, ticker: str) -> PreBreakoutSignal:
    """
    Scan a stock for pre-breakout setup:
    - Currently below EMA50 (hasn't broken out)
    - Price approaching EMA50 (within PROXIMITY_THRESHOLD %)
    - Consolidation pattern (tight range = coiled spring)
    - Volume increasing (accumulation)
    - RSI rising (momentum building)
    - ADX rising (trend forming)
    - Higher lows (bottoming pattern)

    Returns PreBreakoutSignal with score 0-100.
    Score >= 70: APPROACHING (close to breakout)
    Score 45-69: ACCUMULATING (building base)
    Score < 45: NO_SETUP
    """
    result = PreBreakoutSignal(ticker=ticker)

    # Normalize columns
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ('open', 'high', 'low', 'close', 'volume'):
            col_map[c] = cl.capitalize()
    df = df.rename(columns=col_map)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        result.error = f"Missing columns: {missing}"
        return result

    if len(df) < MIN_BARS_REQUIRED:
        result.error = f"Insufficient data: {len(df)} bars"
        return result

    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
    if len(df) < MIN_BARS_REQUIRED:
        result.error = f"Insufficient valid data: {len(df)}"
        return result

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ema = _ema(close, EMA_PERIOD)
    rsi = _rsi(close, 14)
    adx_s, di_plus_s, di_minus_s = _adx(high, low, close, 14)
    atr_s = _atr(high, low, close, 14)
    vol_avg = volume.rolling(VOLUME_AVG_PERIOD).mean()

    # Skip warmup
    if len(df) < EMA_PERIOD + CONSOLIDATION_WINDOW + 5:
        result.error = f"Insufficient data for pre-breakout analysis: {len(df)}"
        return result

    # ── Latest values ─────────────────────────────────────────────────────────
    last_close = float(close.iloc[-1])
    last_ema = float(ema.iloc[-1])
    last_rsi = float(rsi.iloc[-1])
    last_adx = float(adx_s.iloc[-1])
    last_di_plus = float(di_plus_s.iloc[-1])
    last_di_minus = float(di_minus_s.iloc[-1])
    last_atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 0.0
    last_vol = float(volume.iloc[-1])
    last_vol_avg = float(vol_avg.iloc[-1]) if not pd.isna(vol_avg.iloc[-1]) else 0.0
    data_date = str(df.index[-1])[:10] if not isinstance(df.index[-1], (int, float)) else ""

    result.close = last_close
    result.ema_value = last_ema
    result.rsi = last_rsi
    result.adx = last_adx
    result.di_plus = last_di_plus
    result.di_minus = last_di_minus
    result.atr = last_atr
    result.data_date = data_date
    result.volume = last_vol
    result.volume_avg = last_vol_avg

    # ── Must be BELOW EMA50 (pre-breakout, not post) ───────────────────────────
    tolerance = last_ema * EMA_TOLERANCE
    if last_close > last_ema + tolerance:
        # Already above EMA — this is a post-breakout stock, skip
        result.signal = "ALREADY_BREAKOUT"
        result.signal_ar = "اخترق بالفعل"
        return result

    # Distance to EMA50 (%)
    distance_pct = ((last_ema - last_close) / last_ema) * 100 if last_ema > 0 else 999
    result.distance_pct = distance_pct

    # If too far from EMA50, not approaching
    if distance_pct > PROXIMITY_THRESHOLD * 2:
        result.signal = "NO_SETUP"
        return result

    # ── Count candles below EMA (consolidation length) ────────────────────────
    candles_below = 0
    start_bar = EMA_PERIOD
    for i in range(len(df) - 1, start_bar - 1, -1):
        if pd.isna(ema.iloc[i]):
            break
        c = float(close.iloc[i])
        e = float(ema.iloc[i])
        tol = e * EMA_TOLERANCE
        if c < e - tol:
            candles_below += 1
        else:
            break  # stop at first candle above EMA
    result.candles_below = candles_below

    # ── Consolidation tightness (recent range vs ATR) ──────────────────────────
    recent_window = df.iloc[-CONSOLIDATION_WINDOW:]
    recent_high_val = float(recent_window['High'].max())
    recent_low_val = float(recent_window['Low'].min())
    range_pct = ((recent_high_val - recent_low_val) / last_close) * 100 if last_close > 0 else 999
    result.consolidation_pct = range_pct
    result.recent_high = recent_high_val

    # ── Volume trend (recent vs older) ─────────────────────────────────────────
    recent_vol = float(volume.iloc[-VOLUME_TREND_WINDOW:].mean())
    older_vol = float(volume.iloc[-VOLUME_TREND_WINDOW*2:-VOLUME_TREND_WINDOW].mean())
    vol_ratio = recent_vol / older_vol if older_vol > 0 else 1.0
    result.volume_ratio = vol_ratio

    if vol_ratio > 1.2:
        result.volume_trend = "rising"
    elif vol_ratio < 0.8:
        result.volume_trend = "falling"
    else:
        result.volume_trend = "flat"

    # ── RSI trend (rising from low levels) ─────────────────────────────────────
    recent_rsi = float(rsi.iloc[-RSI_RISING_WINDOW:].mean())
    older_rsi = float(rsi.iloc[-RSI_RISING_WINDOW*2:-RSI_RISING_WINDOW].mean())
    rsi_delta = recent_rsi - older_rsi

    if rsi_delta > 3:
        result.rsi_trend = "rising"
    elif rsi_delta < -3:
        result.rsi_trend = "falling"
    else:
        result.rsi_trend = "flat"

    # ── ADX trend (is a trend forming?) ────────────────────────────────────────
    recent_adx = float(adx_s.iloc[-RSI_RISING_WINDOW:].mean())
    older_adx = float(adx_s.iloc[-RSI_RISING_WINDOW*2:-RSI_RISING_WINDOW].mean())
    adx_delta = recent_adx - older_adx

    if adx_delta > 2:
        result.adx_trend = "rising"
    elif adx_delta < -2:
        result.adx_trend = "falling"
    else:
        result.adx_trend = "flat"

    # ── Higher lows detection (last 3 swing lows) ──────────────────────────────
    # Simple: check if last 3 lows are making higher lows
    recent_lows = low.iloc[-CONSOLIDATION_WINDOW:].values
    # Find swing lows (local minima)
    swing_lows = []
    for i in range(2, len(recent_lows) - 2):
        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] \
           and recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]:
            swing_lows.append(recent_lows[i])

    if len(swing_lows) >= 2:
        result.higher_lows = swing_lows[-1] > swing_lows[-2]

    # ── Score Calculation (0-100) ──────────────────────────────────────────────
    # 6 factors, weighted to total 100:

    score = 0

    # 1. Proximity to EMA50 (30 pts) — closer = more points
    #    0% below = 30 pts, 5% below = 15 pts, 10%+ = 0 pts
    if distance_pct <= 1:
        score += 30
    elif distance_pct <= 2:
        score += 25
    elif distance_pct <= 3:
        score += 20
    elif distance_pct <= 5:
        score += 12
    elif distance_pct <= 8:
        score += 5

    # 2. Consolidation tightness (20 pts) — tighter range = more points
    #    <5% range = 20 pts, 5-10% = 15, 10-15% = 8, >15% = 0
    if range_pct < 5:
        score += 20
    elif range_pct < 8:
        score += 15
    elif range_pct < 12:
        score += 10
    elif range_pct < 15:
        score += 5

    # 3. Volume trend (20 pts) — rising volume = accumulation
    if result.volume_trend == "rising":
        score += 20
    elif result.volume_trend == "flat":
        score += 8

    # 4. RSI trend (15 pts) — rising = momentum building
    if result.rsi_trend == "rising":
        score += 15
    elif result.rsi_trend == "flat":
        score += 5

    # 5. ADX trend (10 pts) — rising = trend forming
    if result.adx_trend == "rising":
        score += 10
    elif result.adx_trend == "flat" and last_adx > ADX_FORMING_THRESHOLD:
        score += 5

    # 6. Higher lows (5 pts) — bottoming pattern
    if result.higher_lows:
        score += 5

    result.score = score

    # ── Signal Classification ──────────────────────────────────────────────────
    if score >= 70:
        result.signal = "APPROACHING"
        result.signal_ar = "🔴 قرب اختراق"
        # Risk management: stop below recent low, target at EMA50 + risk
        if last_atr > 0:
            result.stop_loss = recent_low_val - (0.5 * last_atr)
            risk = last_close - result.stop_loss
            result.target = last_ema + risk  # target = EMA50 + risk
        elif recent_low_val > 0:
            result.stop_loss = recent_low_val * 0.98
            risk = last_close - result.stop_loss
            result.target = last_ema + risk
    elif score >= MIN_PRE_BREAKOUT_SCORE:
        result.signal = "ACCUMULATING"
        result.signal_ar = "🟡 تحت التجميع"
    else:
        result.signal = "NO_SETUP"
        result.signal_ar = "لا يوجد نمط"

    return result


def scan_pre_breakout(stock_list, download_func, progress_callback=None):
    """Scan all stocks for pre-breakout setups."""
    results = []
    total = len(stock_list)
    for i, stock_info in enumerate(stock_list, 1):
        ticker = stock_info["symbol"]
        if progress_callback and i % 20 == 0:
            progress_callback(i, total, len(results))
        try:
            df = download_func(ticker, n_bars=250, retries=2)
            if df is None or df.empty:
                continue
            signal = evaluate_pre_breakout(df, ticker)
            # Only keep stocks with a setup (skip NO_SETUP and ALREADY_BREAKOUT)
            if signal.signal in ("APPROACHING", "ACCUMULATING"):
                results.append(signal)
        except Exception as e:
            logger.debug(f"  {ticker}: {str(e)[:60]}")
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    return results


def format_pre_breakout_summary(signals):
    """Format pre-breakout results for Telegram message."""
    approaching = [s for s in signals if s.is_approaching]
    accumulating = [s for s in signals if s.is_accumulating]

    lines = ["📊 *تقرير EGX — الأسهم القريبة من الاختراق*", ""]
    lines.append(f"🔍 تم مسح {len(signals)} سهم | 🔴 {len(approaching)} قرب اختراق | 🟡 {len(accumulating)} تحت التجميع")
    lines.append("")

    if approaching:
        lines.append("🔴 *قرب الاختراق — راقبها عن قرب:*")
        lines.append("")
        for i, s in enumerate(approaching, 1):
            lines.append(f"{i}. *{s.ticker}* — {s.close:.2f} EGP")
            lines.append(f"   📏 المسافة من EMA50: {s.distance_pct:.1f}%")
            lines.append(f"   📊 النطاق: {s.consolidation_pct:.1f}% | شموع تحت EMA: {s.candles_below}")
            lines.append(f"   📦 الحجم: {s.volume_trend} ({s.volume_ratio:.1f}x) | RSI: {s.rsi:.0f} ({s.rsi_trend})")
            lines.append(f"   📈 ADX: {s.adx:.0f} ({s.adx_trend}) | قمم أعلى: {'✅' if s.higher_lows else '❌'}")
            lines.append(f"   🎯 مقاومة: {s.recent_high:.2f} | EMA50: {s.ema_value:.2f}")
            if s.stop_loss > 0:
                lines.append(f"   🛑 وقف: {s.stop_loss:.2f} | 🎯 هدف: {s.target:.2f}")
            lines.append(f"   ⭐ Score: {s.score}/100")
            lines.append("")
    else:
        lines.append("⚪ لا توجد أسهم قريبة من الاختراق اليوم.")
        lines.append("")

    if accumulating:
        lines.append("🟡 *تحت التجميع — استعد للمتابعة:*")
        lines.append("")
        for s in accumulating[:15]:
            emoji = "🔴" if s.score >= 60 else "🟡"
            lines.append(f"   {emoji} {s.ticker} — {s.close:.2f} | dist: {s.distance_pct:.1f}% | "
                        f"score: {s.score}/100 | vol: {s.volume_trend} | RSI: {s.rsi:.0f} {s.rsi_trend}")
        if len(accumulating) > 15:
            lines.append(f"   ... و {len(accumulating) - 15} سهم آخر")
        lines.append("")

    lines += [
        "─────────────────────",
        "📊 الاستراتيجية: Pre-Breakout Scanner",
        f"⚙️ EMA{EMA_PERIOD} | مسافة < {PROXIMITY_THRESHOLD}% | تجميع {CONSOLIDATION_WINDOW} يوم",
        "⚠️ هذه ليست نصيحة استثمارية",
    ]
    return "\n".join(lines)


def format_pre_breakout_detail(signal: PreBreakoutSignal) -> str:
    """Format a single stock's pre-breakout analysis for /stock command."""
    lines = [
        f"📊 *{signal.ticker}* — {signal.signal_ar}",
        "",
        f"💰 السعر: *{signal.close:.2f} EGP*",
        f"📈 EMA{EMA_PERIOD}: {signal.ema_value:.2f}",
        f"📏 المسافة: {signal.distance_pct:.1f}% تحت EMA50",
        f"📅 البيانات: {signal.data_date}",
        "",
    ]

    if signal.error:
        lines.append(f"❌ خطأ: {signal.error}")
        return "\n".join(lines)

    lines.append(f"⭐ Score: *{signal.score}/100*")
    lines.append("")

    lines.append("🔬 *التحليل:*")
    lines.append(f"   📐 النطاق السعري: {signal.consolidation_pct:.1f}% (آخر {CONSOLIDATION_WINDOW} يوم)")
    lines.append(f"   📦 الحجم: {signal.volume_trend} (نسبة {signal.volume_ratio:.1f}x)")
    lines.append(f"   📊 RSI: {signal.rsi:.1f} ({signal.rsi_trend})")
    lines.append(f"   📈 ADX: {signal.adx:.1f} ({signal.adx_trend})")
    lines.append(f"   🔀 +DI: {signal.di_plus:.1f} | -DI: {signal.di_minus:.1f}")
    lines.append(f"   📉 قمم أعلى: {'✅ نعم — قاع صاعد' if signal.higher_lows else '❌ لا'}")
    lines.append(f"   🎯 مقاومة قريبة: {signal.recent_high:.2f}")
    lines.append("")

    lines.append("📋 *Score Breakdown:*")
    lines.append(f"   📏 القرب من EMA50: {'✅' if signal.distance_pct <= 3 else '⚠️' if signal.distance_pct <= 5 else '❌'}")
    lines.append(f"   📐 ضغط النطاق: {'✅' if signal.consolidation_pct < 8 else '⚠️' if signal.consolidation_pct < 12 else '❌'}")
    lines.append(f"   📦 الحجم المتزايد: {'✅' if signal.volume_trend == 'rising' else '⚠️' if signal.volume_trend == 'flat' else '❌'}")
    lines.append(f"   📊 RSI الصاعد: {'✅' if signal.rsi_trend == 'rising' else '⚠️' if signal.rsi_trend == 'flat' else '❌'}")
    lines.append(f"   📈 ADX المتزايد: {'✅' if signal.adx_trend == 'rising' else '⚠️' if signal.adx_trend == 'flat' else '❌'}")
    lines.append("")

    if signal.stop_loss > 0:
        lines.append("⚖️ *إدارة المخاطر:*")
        lines.append(f"   🛑 وقف الخسارة: *{signal.stop_loss:.2f} EGP*")
        lines.append(f"   🎯 الهدف (EMA50): *{signal.target:.2f} EGP*")
        lines.append("")

    lines += [
        "─────────────────────",
        "⚠️ للمعلومات فقط — ليست نصيحة استثمارية",
    ]
    return "\n".join(lines)
