"""
config.py
---------
Central configuration for the EGX Bot scoring engine.

Liquidity-First Strategy v2 — all thresholds and constants in ONE place.
No magic numbers scattered across modules. Every value here was agreed
upon during the strategy review and is tuned for the Egyptian Stock Exchange.

Changes to scoring behavior MUST go through this file.
"""

from __future__ import annotations


# ─── Liquidity Gate (exclusion filter, not a scoring factor) ─────────────────
# Stocks below these thresholds are EXCLUDED entirely — never scored.

#: Minimum daily turnover (value traded) in EGP. Below this = illiquid.
MIN_TURNOVER_EGP: float = 1_000_000

#: Number of trading days to average when checking liquidity.
#: Must sustain the minimum turnover over this window, not just one day.
MIN_TURNOVER_AVG_DAYS: int = 20


# ─── Price Limit Filter (EGX daily circuit breaker) ──────────────────────────
# EGX enforces a ±10% daily price limit. Stocks at the limit are frozen
# — you can't execute at a good price, so we exclude them.

#: Percentage threshold for detecting a daily price limit hit.
#: EGX standard is 10%. If |change_pct| >= this, stock is excluded.
PRICE_LIMIT_THRESHOLD_PCT: float = 10.0


# ─── Volume Surge Confirmation ───────────────────────────────────────────────
# A single day of high volume is noise. We require sustained confirmation.

#: Number of trading days to look back for volume surge baseline.
VOLUME_SURGE_LOOKBACK_DAYS: int = 3

#: Number of consecutive days the surge must persist to be confirmed.
#: Prevents FOMO entry on a one-day spike.
CONFIRMATION_DAYS: int = 2


# ─── Risk Management ─────────────────────────────────────────────────────────

#: Minimum risk/reward ratio. Below this = not worth the risk.
MIN_RISK_REWARD_RATIO: float = 2.0

#: ATR multiplier for stop-loss calculation.
#: Stop = current_price - (ATR × multiplier) for longs.
ATR_STOP_LOSS_MULTIPLIER: float = 1.5


# ─── Momentum / RSI ──────────────────────────────────────────────────────────

#: RSI above this = overbought (bearish signal for new entries).
RSI_OVERBOUGHT: float = 70.0

#: RSI below this = oversold (potential bounce, but weak momentum).
RSI_OVERSOLD: float = 30.0


# ─── Relative Strength vs EGX 30 ─────────────────────────────────────────────

#: Period (trading days) for comparing stock performance vs the index.
RELATIVE_STRENGTH_PERIOD: int = 20


# ─── Scoring Thresholds ──────────────────────────────────────────────────────

#: Score >= this → "Buy" recommendation.
BUY_THRESHOLD: int = 70

#: Score 50-69 → "Watch" recommendation.
WATCH_THRESHOLD: int = 50

#: Score <= this → "Sell" recommendation.
SELL_THRESHOLD: int = 30

#: Minimum data quality (0.0-1.0) required for Buy/Sell recommendations.
MIN_DATA_QUALITY: float = 0.8

#: Minimum data quality for Watch recommendations.
MIN_DATA_QUALITY_WATCH: float = 0.7


# ─── Price Deviation Between Sources ─────────────────────────────────────────

#: If price difference between stockanalysis.com and tvDatafeed
#: exceeds this percentage, stock is marked "Needs Verification"
#: and excluded from scoring.
PRICE_DEVIATION_THRESHOLD_PCT: float = 4.0


# ─── Data Freshness ─────────────────────────────────────────────────────────

#: Data older than this (in hours) is considered stale.
STALE_DATA_HOURS: int = 24

#: Suspicious daily price movement threshold. Above this = possible
#: data error, not a real move. Triggers a flag, not outright rejection.
SUSPICIOUS_PRICE_CHANGE_PCT: float = 20.0
