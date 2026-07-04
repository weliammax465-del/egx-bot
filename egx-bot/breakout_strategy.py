"""
EMA50 Breakout Strategy — Pine Script translation for EGX Bot
===============================================================
Replaces the Liquidity-First v2 scoring engine with a clean
signal-based breakout strategy matching the user's Pine Script.

Strategy flow:
1. Stock trades below EMA50 for N candles (precondition / setup)
2. Breakout candle: close crosses above EMA50 -> record breakout_high
3. waiting_confirmation = True
4. On subsequent candles, check ALL conditions:
   - cond1: candles_below_count >= 2 (was below EMA before breakout)
   - cond2: close > EMA50 (price above EMA)
   - cond3: close > breakout_high (broke above breakout candle's high)
   - cond4: RSI > rsi_threshold (momentum confirmation)
   - cond5: +DI > -DI (bullish directional movement)
   - cond6: ADX > adx_threshold (trend strength)
   - cond7: volume > volume_avg (volume confirmation)
   - cond8: price hasn't returned below EMA (breakout held)
5. ALL conditions met -> BUY signal
6. Price falls below EMA while waiting -> cancel signal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Strategy Parameters ─────────────────────────────────────────────────────

EMA_PERIOD = 50
RSI_THRESHOLD = 50.0
ADX_THRESHOLD = 20.0
VOLUME_AVG_PERIOD = 20
MIN_CANDLES_BELOW = 2      # cond1: at least 2 candles below EMA before breakout
MIN_BARS_REQUIRED = 60     # minimum data points needed (EMA50 + buffer)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class BreakoutSignal:
    """Result of the breakout strategy for a single stock."""
    ticker: str
    signal: str = "NO_SIGNAL"          # "BUY" | "WAIT" | "NO_SIGNAL"
    signal_ar: str = "لا توجد إشارة"

    # All 8 conditions
    cond1: bool = False                 # candles below EMA before breakout
    cond2: bool = False                 # price above EMA
    cond3: bool = False                 # close > breakout_high
    cond4: bool = False                 # RSI > threshold
    cond5: bool = False                 # +DI > -DI
    cond6: bool = False                 # ADX > threshold
    cond7: bool = False                 # volume > average
    cond8: bool = False                 # hasn't returned below EMA

    # Details for display
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

    # Risk management (computed from ATR)
    stop_loss: float = 0.0
    target: float = 0.0
    atr: float = 0.0

    # Metadata
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
            "ticker": self.ticker,
            "signal": self.signal,
            "conditions_met": self.conditions_met,
            "ema": round(self.ema_value, 2),
            "close": round(self.close, 2),
            "breakout_high": round(self.breakout_high, 2),
            "rsi": round(self.rsi, 1),
            "adx": round(self.adx, 1),
            "di_plus": round(self.di_plus, 1),
            "di_minus": round(self.di_minus, 1),
            "volume": int(self.volume),
            "volume_avg": int(self.volume_avg),
            "stop_loss": round(self.stop_loss, 2),
            "target": round(self.target, 2),
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

    # ── Compute Indicators ────────────────────────────────────────────────────
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    ema = _ema(close, EMA_PERIOD)
    rsi = _rsi(close, 14)
    adx_s, di_plus_s, di_minus_s = _adx(high, low, close, 14)
    atr_s = _atr(high, low, close, 14)
    vol_avg = volume.rolling(VOLUME_AVG_PERIOD).mean()

    # ── Track Breakout State (walk through candles) ───────────────────────────
    candles_below_count = 0
    breakout_high = 0.0
    breakout_date = ""
    breakout_idx = -1
    waiting = False
    returned_below = False
    highest_since_breakout = 0.0
    pre_breakout_count = 0  # candles below EMA before the breakout

    for i in range(len(df)):
        if pd.isna(ema.iloc[i]):
            continue

        c = float(close.iloc[i])
        h = float(high.iloc[i])
        e = float(ema.iloc[i])

        if c < e:
            # Below EMA
            if not waiting:
                candles_below_count += 1
            else:
                # Price returned below EMA while waiting -> cancel
                returned_below = True
                waiting = False
                pre_breakout_count = candles_below_count  # save for cond1
                candles_below_count = 1  # reset
        elif c > e:
            # Above EMA
            if not waiting and candles_below_count >= MIN_CANDLES_BELOW:
                # Breakout candle!
                breakout_high = h
                breakout_date = str(df.index[i])[:10] if not isinstance(df.index[i], (int, float)) else str(i)
                breakout_idx = i
                pre_breakout_count = candles_below_count
                waiting = True
                highest_since_breakout = h
                candles_below_count = 0
            elif waiting:
                if h > highest_since_breakout:
                    highest_since_breakout = h

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

    # Fill result
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
    # cond1: At least MIN_CANDLES_BELOW candles were below EMA before breakout
    result.cond1 = (pre_breakout_count >= MIN_CANDLES_BELOW) if breakout_idx >= 0 else (candles_below_count >= MIN_CANDLES_BELOW)

    # cond2: Price is above EMA
    result.cond2 = last_close > last_ema

    # cond3: Close > breakout_high
    result.cond3 = waiting and (last_close > breakout_high) if breakout_high > 0 else False

    # cond4: RSI > threshold
    result.cond4 = last_rsi > RSI_THRESHOLD

    # cond5: +DI > -DI
    result.cond5 = last_di_plus > last_di_minus

    # cond6: ADX > threshold
    result.cond6 = last_adx > ADX_THRESHOLD

    # cond7: Volume > average
    result.cond7 = last_vol > last_vol_avg if last_vol_avg > 0 else False

    # cond8: Price hasn't returned below EMA
    result.cond8 = waiting and not returned_below

    # ── Final Signal ──────────────────────────────────────────────────────────
    all_met = (result.cond1 and result.cond2 and result.cond3 and result.cond4
               and result.cond5 and result.cond6 and result.cond7 and result.cond8)

    if all_met:
        result.signal = "BUY"
        result.signal_ar = "🟢 شراء"
        # Risk management: ATR-based
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


# ─── Formatting for Telegram ─────────────────────────────────────────────────

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
        for s in wait_signals[:10]:
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
