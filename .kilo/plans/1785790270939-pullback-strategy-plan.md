# EGX Pullback Entry Strategy — Implementation Plan

## Problem Analysis

The current Pre-Breakout Scanner finds stocks *approaching* EMA50 but most never break out. The EMA50 Breakout strategy (8/8 conditions) is too strict and misses valid setups. The scoring v2 system exists but is not used in the scheduled Telegram report. The result: too many false signals, stocks that spike up then reverse, and user frustration with unreliable recommendations.

## New Strategy: Pullback Entry (Post-Breakout Confirmation)

Instead of guessing which stocks *will* break out, this strategy waits for the breakout to happen, then enters on the pullback to support. This is the highest-probability setup in technical analysis.

### Strategy Logic (5 phases)

**Phase 1 — Breakout Detection**
- Stock must have broken above EMA50 in the last 5–15 trading days
- The breakout candle must have volume > 1.5× the 20-day average
- The breakout must hold (price hasn't returned below EMA50 since)

**Phase 2 — Pullback to Support**
- Price has pulled back to EMA20 (primary) or EMA50 (secondary)
- The pullback shows declining volume (distribution is over)
- RSI has recovered from oversold territory (> 40) or is holding above 50
- Price is still above EMA50 (the breakout level is intact)

**Phase 3 — Bounce Confirmation**
- Latest candle shows a bullish close (close > open)
- Volume on the bounce is increasing (vs pullback days)
- ADX > 20 (trend is still forming, not collapsing)
- MACD histogram is expanding (momentum returning)

**Phase 4 — Entry Signal**
- Entry at or near EMA20 (the pullback support)
- Stop-loss: below the recent pullback low (or EMA50 if below recent low)
- Target: entry + (2.0 × risk) — minimum 2:1 R/R

**Phase 5 — Quality Filters (all must pass)**
- Minimum daily turnover ≥ 1M EGP (liquidity gate)
- Price deviation < 4% between sources (data quality)
- R/R ratio ≥ 2.0 (risk management)
- ADX ≥ 20 (trend strength)
- Stock not at daily price limit (±10%)
- Data freshness: live or same-day

### Scoring (0–100, weighted)

| Factor | Weight | Description |
|--------|--------|-------------|
| Trend Alignment | 30% | EMA stack (price > EMA20 > EMA50), ADX strength, MACD alignment |
| Pullback Quality | 25% | How clean the pullback is (declining vol, RSI recovery, higher lows) |
| Volume Confirmation | 20% | Bounce volume > pullback volume, overall volume trend |
| Risk/Reward | 25% | R/R ratio quality (≥2:1 = max, <1.5:1 = fail) |

**Recommendation thresholds:**
- Score ≥ 75 → **Buy** (strong setup)
- Score 55–74 → **Watch** (monitor for entry)
- Score < 55 → **No Trade** (skip)

## Files to Create

### 1. `egx-bot/pullback_strategy.py` (NEW — core strategy engine)

Contains:
- `PullbackSignal` dataclass (ticker, score, entry, stop_loss, target, rr_ratio, phase, indicators, reasons)
- `detect_breakout(df, ticker)` — finds recent EMA50 breakouts (last 15 days)
- `evaluate_pullback(df, ticker)` — main function: checks pullback + bounce conditions
- `scan_pullback_stocks(stock_list, download_func)` — scans all stocks, returns pullback signals
- `format_pullback_summary(signals)` — Telegram message formatting
- `format_pullback_detail(signal)` — per-stock detail formatting
- `score_pullback(signal)` — computes the 0–100 weighted score

### 2. `egx-bot/strategy_config.py` (NEW — strategy-specific config)

All thresholds in one place:
- `BREAKOUT_LOOKBACK_DAYS = 15` — how far back to look for breakout
- `BREAKOUT_VOLUME_MULTIPLIER = 1.5` — min volume on breakout candle
- `PULLBACK_EMA_PRIMARY = 20` — primary pullback support EMA
- `PULLBACK_EMA_SECONDARY = 50` — secondary pullback support EMA
- `PULLBACK_MAX_DISTANCE_PCT = 10.0` — max % pullback from breakout high
- `BOUNCE_VOLUME_MIN = 1.2` — bounce vol must exceed pullback vol by 20%
- `RSI_PULLBACK_MIN = 40` — RSI must recover above this on pullback
- `ADX_TREND_MIN = 20` — minimum ADX for trend strength
- `MIN_RR_RATIO = 2.0` — minimum risk/reward
- `BUY_THRESHOLD = 75` — score ≥ this → Buy
- `WATCH_THRESHOLD = 55` — score ≥ this → Watch
- `MAX_CANDLES_BELOW_EMA50_AFTER_BREAKOUT = 5` — price can't be below EMA50 for more than 5 candles after breakout

## Files to Modify

### 3. `egx-bot/bot.py` — Integrate pullback strategy into scheduled report

Changes:
- Import `scan_pullback_stocks`, `format_pullback_summary`, `format_pullback_detail` from `pullback_strategy`
- Replace `scan_pre_breakout` + `find_5star_stocks` with `scan_pullback_stocks` in `send_scheduled_report()`
- Update `/today` and `/watchlist` commands to use pullback strategy
- Update `/stock` command to show pullback detail when stock is in pullback phase
- Keep `find_5star_stocks` as a fallback option (not primary)

### 4. `egx-bot/scoring.py` — Add pullback scoring function

Add `compute_pullback_score(signal)` that uses the 4-factor weighted scoring from the plan. Keep existing `compute_score_v2` for backward compatibility.

## Integration: Init In Workflow

The workflow flow becomes:

```
Init In
  → scrape_egx_stock_list()          # Fetch live prices
  → download_stock_history(ticker)    # Get OHLCV data
  → evaluate_pullback(df, ticker)     # Apply pullback strategy
  → score_pullback(signal)            # Score 0-100
  → Filter: score >= WATCH_THRESHOLD  # Only actionable stocks
  → format_pullback_summary()         # Format for Telegram
  → Send via Telegram bot
```

This replaces the current flow:
```
Init In
  → scrape_egx_stock_list()
  → scan_pre_breakout()               # Pre-breakout (speculative)
  → find_5star_stocks()               # 5-star rating
  → Send via Telegram bot
```

## Validation Plan

1. **Backtest**: Run `scan_pullback_stocks()` on historical data (last 30 days) and compare signals vs actual price movement
2. **Dry run**: Deploy the new strategy alongside the old one for 1 week, compare signal quality
3. **Metrics to track**: Signal accuracy (did the stock go up 3%+ within 5 days?), false positive rate, average R/R ratio, win rate
4. **Threshold tuning**: If win rate < 70%, tighten `PULLBACK_MAX_DISTANCE_PCT` or raise `BUY_THRESHOLD`

## Key Design Decisions

1. **Pullback over breakout**: Waiting for confirmation (pullback) is more reliable than catching breakouts early. This is the single biggest improvement.
2. **EMA20 as primary support**: EMA20 is the most commonly watched short-term support in EGX trading. EMA50 is the breakout level itself.
3. **Volume declining on pullback**: This distinguishes healthy profit-taking from distribution. If volume rises on the pullback, it's a distribution signal — exclude.
4. **5-phase quality gate**: Each phase filters out progressively weaker setups. A stock must pass all 5 phases to get a Buy signal.
5. **Weighted scoring over binary conditions**: The current 8/8 binary approach is too rigid. Weighted scoring allows nuanced evaluation where strong indicators compensate for slightly weaker ones.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Fewer signals (more selective) | This is intentional — quality over quantity. The strategy should produce 3–8 Buy signals per week, not 30+ |
| Pullback never comes after breakout | The `PULLBACK_MAX_DISTANCE_PCT` cap ensures we don't wait forever; stocks that don't pullback within 15 days are skipped |
| EMA20 acts as resistance (not support) | The bounce confirmation phase catches this — if price can't hold above EMA20 on the bounce, the signal is downgraded or excluded |
| Data freshness issues | The existing `validate_scraped_price` and `check_scrape_freshness` gates already handle this |

## Open Questions

1. Should the `/stock` command show both pre-breakout AND pullback analysis for the same stock? (Yes — show whichever phase the stock is currently in)
2. Should the scheduled report send individual messages per stock (like 5-star format) or a single summary? (Individual messages — easier to act on)
3. Should we keep the old pre-breakout scanner as a fallback command? (Yes — keep `/today` as pullback, add `/prebreakout` for the old scanner)
