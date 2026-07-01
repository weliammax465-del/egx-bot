"""
stock_scanner.py
----------------
Scans ALL EGX stocks with real-time prices and professional technical analysis.

Pipeline:
  1. Data Collection: Scrape stockanalysis.com → fallback to egx_stocks.json
  2. Data Validation: Validate symbols, prices, OHLCV data
  3. Historical Data: Download from TradingView (with retries)
  4. Technical Analysis: Compute 15+ indicators
  5. Scoring: Deterministic 0-100 score with recommendation
  6. Ranking: Sort by score, filter for quality
"""

from __future__ import annotations

import os
import json
import logging
import time
from typing import Optional
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from indicators import StockAnalysis, analyze_stock
from scoring import compute_score
from data.symbols import normalize_symbol, is_valid_egx_symbol, get_arabic_name, get_canonical_name, update_canonical_list
from data.validator import (
    validate_ohlcv, validate_price, validate_scraped_price,
    classify_scrape_error, check_scraper_freshness, deduplicate_stocks,
    ScanStatus,
)

logger = logging.getLogger(__name__)

STOCKANALYSIS_URL = "https://stockanalysis.com/list/egyptian-stock-exchange/"
LAST_REPORT_FILE = os.path.join(os.path.dirname(__file__), ".egx_last_report.json")

# Module-level scan status — accessible after scan_all_stocks()
last_scan_status = ScanStatus()
FALLBACK_STOCKS_FILE = os.path.join(os.path.dirname(__file__), "egx_stocks.json")


# ─── Data Collection ─────────────────────────────────────────────────────────

def _load_fallback_stock_list() -> list[dict]:
    """Load saved stock list as fallback."""
    try:
        with open(FALLBACK_STOCKS_FILE, "r", encoding="utf-8") as f:
            stocks = json.load(f)
            logger.info(f"Loaded {len(stocks)} stocks from fallback file.")
            return stocks
    except Exception as e:
        logger.error(f"Failed to load fallback stock list: {e}")
        return []


def scrape_egx_stock_list() -> list[dict]:
    """
    Scrape the full EGX stock list from stockanalysis.com.
    Validates every symbol against the canonical EGX list.
    Falls back to egx_stocks.json if scraping fails.
    Classifies errors precisely: page structure change, timeout, no response, etc.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(STOCKANALYSIS_URL, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        error_type = classify_scrape_error(e, "stockanalysis.com")
        logger.error(f"stockanalysis.com scrape failed — error type: {error_type} | {e}")
        last_scan_status.failed_sources.append(f"stockanalysis.com ({error_type})")
        last_scan_status.used_fallback = True
        last_scan_status.limited_coverage = True
        fallback = _load_fallback_stock_list()
        last_scan_status.coverage_count = len(fallback)
        return fallback

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        logger.error("stockanalysis.com: page structure changed — no <table> found")
        last_scan_status.failed_sources.append("stockanalysis.com (page_structure_changed: no table)")
        last_scan_status.used_fallback = True
        last_scan_status.limited_coverage = True
        fallback = _load_fallback_stock_list()
        last_scan_status.coverage_count = len(fallback)
        return fallback

    tbody = table.find("tbody")
    if not tbody:
        logger.error("stockanalysis.com: page structure changed — no <tbody> found")
        last_scan_status.failed_sources.append("stockanalysis.com (page_structure_changed: no tbody)")
        last_scan_status.used_fallback = True
        last_scan_status.limited_coverage = True
        fallback = _load_fallback_stock_list()
        last_scan_status.coverage_count = len(fallback)
        return fallback

    raw_stocks = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 5:
            symbol = normalize_symbol(cells[1].get_text(strip=True))
            name = cells[2].get_text(strip=True)
            price_str = cells[4].get_text(strip=True)
            change_str = cells[5].get_text(strip=True)
            market_cap_str = cells[3].get_text(strip=True) if len(cells) > 3 else ""

            try:
                price = float(price_str.replace(",", ""))
            except (ValueError, TypeError):
                price = 0.0

            change_pct = 0.0
            try:
                change_clean = change_str.replace("%", "").replace("+", "").strip()
                if change_clean and change_clean != "-":
                    change_pct = float(change_clean)
                    if change_str.strip().startswith("-"):
                        change_pct = -change_pct
            except (ValueError, TypeError):
                pass

            if symbol and symbol != "NO.":
                # Rigorous price validation — reject zero/null/NaN
                price_result = validate_scraped_price(price, ticker=symbol)
                if price_result.is_valid:
                    raw_stocks.append({
                        "symbol": symbol,
                        "name": name,
                        "price": price,
                        "change_pct": change_pct,
                        "market_cap_str": market_cap_str,
                    })
                else:
                    logger.warning(f"  ⚠️ {symbol}: {price_result.reason}")
                    last_scan_status.total_rejected += 1
                    if price_result.is_suspicious:
                        last_scan_status.suspicious_count += 1
                        last_scan_status.suspicious_tickers.append(symbol)

    # Deduplicate
    raw_stocks = deduplicate_stocks(raw_stocks)

    # Update canonical list with freshly scraped symbols
    update_canonical_list(raw_stocks)

    if not raw_stocks:
        logger.warning("Scraping returned 0 valid stocks. Using fallback.")
        last_scan_status.used_fallback = True
        last_scan_status.limited_coverage = True
        fallback = _load_fallback_stock_list()
        last_scan_status.coverage_count = len(fallback)
        return fallback

    logger.info(f"Scraped {len(raw_stocks)} valid EGX stocks from stockanalysis.com")
    return raw_stocks


# ─── TradingView Historical Data ─────────────────────────────────────────────

_tv_instance = None


def _get_tv():
    """Lazy-initialize TradingView datafeed."""
    global _tv_instance
    if _tv_instance is None:
        from tvDatafeed import TvDatafeed, Interval
        _tv_instance = TvDatafeed()
    return _tv_instance


def download_stock_history(ticker: str, n_bars: int = 250, retries: int = 2) -> Optional[pd.DataFrame]:
    """Download historical OHLCV from TradingView with retry logic."""
    from tvDatafeed import TvDatafeed, Interval
    tv = _get_tv()

    for attempt in range(retries + 1):
        try:
            df = tv.get_hist(symbol=ticker, exchange="EGX", interval=Interval.in_daily, n_bars=n_bars)
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    "open": "Open", "high": "High", "low": "Low",
                    "close": "Close", "volume": "Volume",
                })
                if "symbol" in df.columns:
                    df = df.drop(columns=["symbol"])
                return df
            if attempt < retries:
                time.sleep(1 * (attempt + 1))
            else:
                return None
        except Exception as e:
            error_type = classify_scrape_error(e, "TradingView")
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
            else:
                logger.warning(f"TradingView failed for {ticker} after {retries+1} attempts — error type: {error_type} | {str(e)[:80]}")
                if error_type == "insufficient_data" or (df is not None and len(df) < 50):
                    logger.info(f"  {ticker}: insufficient historical data ({len(df) if df is not None else 0} bars, need 50)")
                return None
    return None


# ─── Scan Caching ────────────────────────────────────────────────────────────

_scan_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


def _get_cached_scan() -> list[StockAnalysis] | None:
    """Return cached scan results if fresh enough."""
    if _scan_cache and time.time() - _scan_cache.get("time", 0) < _CACHE_TTL:
        logger.info("Returning cached scan results.")
        return _scan_cache.get("data")
    return None


def _set_cached_scan(data: list[StockAnalysis]) -> None:
    """Cache scan results."""
    _scan_cache["time"] = time.time()
    _scan_cache["data"] = data


# ─── EGX 30 Index: 20-day change for Relative Strength filter ────────────────

def get_egx30_change_20d() -> float | None:
    """
    Fetch EGX 30 index 20-day % change from TradingView.
    Used as the benchmark for the Relative Strength filter (step 5).
    Returns None if data is unavailable.
    """
    try:
        from tvDatafeed import TvDatafeed, Interval
        tv = _get_tv()
        df = tv.get_hist(symbol="EGX30", exchange="EGX", interval=Interval.in_daily, n_bars=30)
        if df is None or len(df) < 22:
            return None
        df = df.rename(columns={"close": "Close"})
        price_now = float(df["Close"].iloc[-1])
        price_20d = float(df["Close"].iloc[-21])
        if price_20d <= 0:
            return None
        return round((price_now - price_20d) / price_20d * 100, 2)
    except Exception as e:
        logger.warning(f"Could not fetch EGX30 for RS filter: {e}")
        return None


# ─── Single Stock Scan ───────────────────────────────────────────────────────

def scan_single_stock(ticker: str) -> StockAnalysis | None:
    """
    Scan a single stock by ticker — much faster than scan_all_stocks().
    Used by /stock SYMBOL command.
    Applies the full v2 pipeline (gates → indicators → v2 score).
    """
    from filters import (
        pass_all_gates, volume_surge_check, confirmation_check,
        trend_strength_filter, risk_filter, get_exclusion_code,
    )
    from scoring import compute_score_v2
    import config

    ticker = normalize_symbol(ticker)
    if not is_valid_egx_symbol(ticker):
        logger.warning(f"Symbol {ticker} is not a verified EGX symbol.")
        return None

    name_ar = get_arabic_name(ticker)
    name_en = get_canonical_name(ticker) or name_ar

    # Check cache first
    cached = _get_cached_scan()
    if cached:
        for s in cached:
            if s.ticker == ticker:
                return s

    # Download historical data
    df = download_stock_history(ticker, n_bars=250, retries=2)
    if df is None or len(df) < 50:
        logger.warning(f"No data for {ticker}.")
        return None

    # Validate OHLCV
    validation = validate_ohlcv(df, ticker)
    if not validation.is_valid:
        logger.warning(f"Data validation failed for {ticker}: {validation.issues[:2]}")
        return None

    # Run full indicator analysis
    try:
        analysis = analyze_stock(df, ticker, name_en, name_ar)
        analysis.data_freshness = validation.freshness
        analysis.data_quality = validation.quality_score
        analysis.timestamp = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return None

    # v2 pipeline — gates + filters + scoring
    egx30_20d = get_egx30_change_20d()

    # Gates (liquidity + price limit)
    passed_gates, gate_results = pass_all_gates(df, analysis.daily_change_pct, ticker)
    if not passed_gates:
        analysis.scoring_result = _make_excluded_result(get_exclusion_code(gate_results))
        return analysis

    # Volume surge + confirmation
    confirmation = confirmation_check(df, ticker)
    if not confirmation.confirmed:
        analysis.scoring_result = _make_excluded_result(confirmation.exclusion_code or "لا يوجد تأكيد يومين")
        return analysis

    # Trend & Relative Strength
    ema50_ind = next((i for i in analysis.indicators if i.name == "EMA 50"), None)
    rsi_ind   = next((i for i in analysis.indicators if i.name == "RSI"), None)
    ema50 = ema50_ind.value if ema50_ind else 0.0
    rsi   = rsi_ind.value   if rsi_ind   else 50.0

    trend = trend_strength_filter(df, ema50, rsi, egx30_20d, ticker)
    if not trend.passed:
        analysis.scoring_result = _make_excluded_result(trend.exclusion_code)
        return analysis

    # Risk management
    atr_ind = next((i for i in analysis.indicators if i.name == "ATR"), None)
    atr = atr_ind.value if atr_ind else 0.0
    risk = risk_filter(analysis.current_price, atr, analysis.resistance, ticker)
    if not risk.passed:
        analysis.scoring_result = _make_excluded_result(risk.exclusion_code)
        return analysis

    # Compute v2 score
    avg_turnover = df["Volume"].tail(20).mean() * analysis.current_price
    latest_turnover = float(df["Volume"].iloc[-1]) * analysis.current_price
    stock_20d_change = ((analysis.current_price - float(df["Close"].iloc[-21])) / float(df["Close"].iloc[-21]) * 100) if len(df) >= 21 else 0.0

    scoring = compute_score_v2(
        analysis,
        avg_turnover_egp=avg_turnover,
        latest_turnover_egp=latest_turnover,
        stock_change_20d=stock_20d_change,
        egx30_change_20d=egx30_20d if egx30_20d is not None else 0.0,
        confirmation=confirmation,
        rr_ratio=risk.details["rr_ratio"],
        stop_loss=risk.details["stop_loss"],
        target=risk.details["target"],
        data_freshness=validation.freshness,
        data_quality=validation.quality_score,
    )
    analysis.scoring_result = scoring
    return analysis


def _make_excluded_result(exclusion_code: str):
    """Create a ScoringResult for an excluded stock (failed a filter)."""
    from scoring import ScoringResult
    return ScoringResult(
        total_score=0,
        recommendation="No Trade",
        recommendation_ar="لا تداول ⚪",
        exclusion_reason=exclusion_code,
        exclusion_code=exclusion_code,
    )


# ─── Full Scan ───────────────────────────────────────────────────────────────

def scan_all_stocks() -> list[StockAnalysis]:
    """
    Full Liquidity-First v2 pipeline:
    1. Check cache (return if fresh)
    2. Collect stock list (scrape + validate)
    3. Download EGX 30 index for Relative Strength benchmark
    4. Per stock:
       a. Download historical OHLCV (TradingView, with retries)
       b. Price deviation check (>4% → Needs Verification)
       c. OHLCV validation
       d. Liquidity gate + Price limit gate (pre-indicator)
       e. Technical indicators (15+)
       f. Volume surge + 2-day confirmation
       g. Trend & Relative Strength filter
       h. Risk management filter (ATR stop-loss, R/R >= 2:1)
       i. Liquidity-First v2 scoring (4 factors, 0-100)
    5. Sort by score (scored stocks first, excluded after)
    6. Cache and return
    """
    from filters import (
        pass_all_gates, volume_surge_check, confirmation_check,
        trend_strength_filter, risk_filter, get_exclusion_code,
    )
    from scoring import compute_score_v2
    import config

    # Check cache first
    cached = _get_cached_scan()
    if cached is not None:
        return cached

    stock_list = scrape_egx_stock_list()
    if not stock_list:
        logger.error("No stocks available. All sources failed.")
        return []

    total = len(stock_list)
    logger.info(f"Starting v2 scan of {total} EGX stocks...")

    # Fetch EGX 30 benchmark once (used for relative strength filter)
    egx30_20d = get_egx30_change_20d()
    if egx30_20d is not None:
        logger.info(f"EGX 30: 20-day change = {egx30_20d:+.2f}%")
    else:
        logger.warning("EGX 30 data unavailable — RS filter will be skipped")

    results: list[StockAnalysis] = []
    success_count = 0
    fail_count = 0
    rejected_count = 0
    gate_excluded = 0
    filter_excluded = 0

    for i, stock_info in enumerate(stock_list, 1):
        ticker   = stock_info["symbol"]
        name_en  = stock_info["name"]
        live_price = stock_info["price"]
        live_change = stock_info["change_pct"]

        if not is_valid_egx_symbol(ticker):
            logger.debug(f"  ❌ {ticker}: not a verified EGX symbol, skipping")
            rejected_count += 1
            continue

        name_ar = get_arabic_name(ticker, name_en)

        if i % 20 == 0:
            logger.info(
                f"Progress: {i}/{total} | scored={success_count} "
                f"failed={fail_count} rejected={rejected_count} "
                f"gate_excluded={gate_excluded} filter_excluded={filter_excluded}"
            )

        # ── Step 4a: Download historical OHLCV ──
        df = download_stock_history(ticker, n_bars=250, retries=2)

        # ── Step 4b: Price deviation check ──
        if df is not None and len(df) > 0 and live_price > 0:
            last_close = float(df["Close"].iloc[-1])
            price_result = validate_scraped_price(live_price, last_known_price=last_close, ticker=ticker)
            if not price_result.is_valid:
                logger.warning(f"  ⚠️ {ticker}: {price_result.reason}")
                rejected_count += 1
                continue
            elif price_result.is_suspicious:
                logger.warning(f"  🔍 {ticker}: {price_result.reason} — Needs Verification")
                last_scan_status.needs_verification.append(ticker)
                continue

        # ── Step 4c: OHLCV validation ──
        if df is None or len(df) < 50:
            fail_count += 1
            last_scan_status.no_indicators_tickers.append(ticker)
            logger.debug(f"  ⚠️ {ticker}: no history ({len(df) if df is not None else 0} bars)")
            continue

        validation = validate_ohlcv(df, ticker)
        if not validation.is_valid:
            logger.debug(f"  ⚠️ {ticker}: OHLCV validation failed — {validation.issues[0] if validation.issues else 'unknown'}")
            fail_count += 1
            last_scan_status.no_indicators_tickers.append(ticker)
            continue

        freshness = validation.freshness
        data_quality = validation.quality_score

        # ── Step 4d: Liquidity gate + Price limit gate (BEFORE indicators) ──
        passed_gates, gate_results = pass_all_gates(df, live_change, ticker)
        if not passed_gates:
            excl_code = get_exclusion_code(gate_results)
            logger.info(f"  🚫 {ticker}: gate excluded — {excl_code}")
            gate_excluded += 1
            last_scan_status.total_rejected += 1
            # Track in scan results as excluded (for analytics/RecommendationHistory)
            _record_excluded(ticker, live_price, excl_code)
            continue

        # ── Step 4e: Technical indicators ──
        try:
            analysis = analyze_stock(df, ticker, name_en, name_ar)
            if live_price > 0:
                analysis.current_price = live_price
                analysis.daily_change_pct = live_change
            analysis.data_freshness = freshness
            analysis.data_quality = data_quality
            analysis.timestamp = datetime.now().isoformat()
        except Exception as e:
            logger.warning(f"  ❌ {ticker}: indicator analysis failed: {str(e)[:80]}")
            fail_count += 1
            last_scan_status.no_indicators_tickers.append(ticker)
            continue

        # ── Step 4f: Volume surge + 2-day confirmation ──
        confirmation = confirmation_check(df, ticker)
        if not confirmation.confirmed:
            excl_code = confirmation.exclusion_code or "لا يوجد تأكيد يومين"
            logger.info(f"  🚫 {ticker}: confirmation failed — {excl_code}")
            filter_excluded += 1
            analysis.scoring_result = _make_excluded_result(excl_code)
            results.append(analysis)  # keep in results for display, just excluded from recommendations
            continue

        # ── Step 4g: Trend & Relative Strength ──
        ema50_ind = next((ind for ind in analysis.indicators if ind.name == "EMA 50"), None)
        rsi_ind   = next((ind for ind in analysis.indicators if ind.name == "RSI"), None)
        ema50 = ema50_ind.value if ema50_ind else 0.0
        rsi   = rsi_ind.value   if rsi_ind   else 50.0

        trend = trend_strength_filter(df, ema50, rsi, egx30_20d, ticker)
        if not trend.passed:
            logger.info(f"  🚫 {ticker}: trend/RS failed — {trend.exclusion_code}")
            filter_excluded += 1
            analysis.scoring_result = _make_excluded_result(trend.exclusion_code)
            results.append(analysis)
            continue

        # ── Step 4h: Risk management filter ──
        atr_ind = next((ind for ind in analysis.indicators if ind.name == "ATR"), None)
        atr = atr_ind.value if atr_ind else 0.0
        risk = risk_filter(analysis.current_price, atr, analysis.resistance, ticker)
        if not risk.passed:
            logger.info(f"  🚫 {ticker}: risk/reward failed — {risk.exclusion_code}")
            filter_excluded += 1
            analysis.scoring_result = _make_excluded_result(risk.exclusion_code)
            results.append(analysis)
            continue

        # ── Step 4i: Liquidity-First v2 score ──
        avg_turnover    = float(df["Volume"].tail(20).replace(0, float("nan")).mean() * analysis.current_price)
        latest_turnover = float(df["Volume"].iloc[-1]) * analysis.current_price

        stock_20d_change = 0.0
        if len(df) >= 21:
            price_20d = float(df["Close"].iloc[-21])
            if price_20d > 0:
                stock_20d_change = (analysis.current_price - price_20d) / price_20d * 100

        scoring = compute_score_v2(
            analysis,
            avg_turnover_egp=avg_turnover,
            latest_turnover_egp=latest_turnover,
            stock_change_20d=stock_20d_change,
            egx30_change_20d=egx30_20d if egx30_20d is not None else 0.0,
            confirmation=confirmation,
            rr_ratio=risk.details["rr_ratio"],
            stop_loss=risk.details["stop_loss"],
            target=risk.details["target"],
            data_freshness=freshness,
            data_quality=data_quality,
        )
        analysis.scoring_result = scoring

        results.append(analysis)
        success_count += 1

        time.sleep(0.15)

    # Sort: scored stocks (recommendation != No Trade/Excluded) first, by score desc
    def sort_key(s):
        sr = s.scoring_result
        if sr is None:
            return (2, 0)
        if sr.recommendation in ("Buy", "Watch"):
            return (0, -sr.total_score)
        if sr.recommendation == "Sell":
            return (1, -sr.total_score)
        return (2, 0)

    results.sort(key=sort_key)
    _set_cached_scan(results)

    # Update scan status
    last_scan_status.total_scraped = total
    last_scan_status.total_validated = success_count
    last_scan_status.coverage_count = success_count + filter_excluded
    last_scan_status.has_reliable_data = success_count > 0
    last_scan_status.source = "stockanalysis.com" if not last_scan_status.used_fallback else "egx_stocks.json (fallback)"

    logger.info(
        f"v2 Scan complete: {success_count} scored, {filter_excluded} filter-excluded, "
        f"{gate_excluded} gate-excluded, {fail_count} failed, {rejected_count} symbol-rejected"
    )

    return results


def _record_excluded(ticker: str, price: float, exclusion_code: str) -> None:
    """
    Log gate-excluded stocks for potential RecommendationHistory tracking.
    Gate-excluded stocks didn't get indicators computed, so we just log them.
    """
    logger.info(f"  📋 {ticker} gate-excluded (price={price:.2f}, reason={exclusion_code})")


# ─── Last Report Cache (fallback when today's scan fails) ────────────────────

def save_last_report(report_data: dict) -> None:
    """Save the last successful report data for fallback use."""
    try:
        with open(LAST_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved last report cache to {LAST_REPORT_FILE}")
    except Exception as e:
        logger.warning(f"Could not save last report cache: {e}")


def load_last_report() -> dict | None:
    """Load the last successful report for fallback."""
    try:
        with open(LAST_REPORT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded last report cache from {LAST_REPORT_FILE} (date: {data.get('date', 'unknown')})")
            return data
    except FileNotFoundError:
        logger.info("No last report cache found.")
        return None
    except Exception as e:
        logger.warning(f"Could not load last report cache: {e}")
        return None


def get_scan_status() -> ScanStatus:
    """Return the status of the last scan operation."""
    return last_scan_status

# ─── Ranking Helpers ─────────────────────────────────────────────────────────

def get_top_bullish(stocks: list[StockAnalysis], top_n: int = 10) -> list[StockAnalysis]:
    """Top N stocks with strongest bullish signals AND acceptable data quality."""
    bullish = [s for s in stocks if s.composite_score >= 2 and s.data_quality >= 0.5]
    return bullish[:top_n]


def get_top_bearish(stocks: list[StockAnalysis], top_n: int = 5) -> list[StockAnalysis]:
    """Top N stocks with strongest bearish signals AND acceptable data quality."""
    bearish = [s for s in stocks if s.composite_score <= -2 and s.data_quality >= 0.5]
    bearish.sort(key=lambda x: x.composite_score)
    return bearish[:top_n]


def get_buy_signals(stocks: list[StockAnalysis]) -> list[StockAnalysis]:
    """Stocks with Buy recommendation from the scoring engine."""
    return [s for s in stocks if s.scoring_result is not None and s.scoring_result.recommendation == "Buy"]


def get_watchlist(stocks: list[StockAnalysis]) -> list[StockAnalysis]:
    """Stocks with Watch or Buy recommendation."""
    result = []
    for s in stocks:
        if s.scoring_result is not None:
            if s.scoring_result.recommendation in ("Buy", "Watch"):
                result.append(s)
    return result


def get_single_stock(stocks: list[StockAnalysis], ticker: str) -> Optional[StockAnalysis]:
    """Find a specific stock by ticker (case-insensitive)."""
    ticker = normalize_symbol(ticker)
    for s in stocks:
        if s.ticker == ticker:
            return s
    return None


# ─── Formatting ──────────────────────────────────────────────────────────────

def format_analysis_for_ai(stocks: list[StockAnalysis], market_text: str = "") -> str:
    """Format computed analysis data for AI explanation (not generation)."""
    lines = ["EGX STOCK TECHNICAL ANALYSIS — COMPUTED DATA", "=" * 50, ""]

    if market_text:
        lines.append("MARKET INDEX DATA:")
        lines.append(market_text)
        lines.append("")

    analyzed = [s for s in stocks if s.indicators and s.data_quality >= 0.5]
    total_with_data = len(analyzed)
    total_all = len(stocks)

    buy_count = sum(1 for s in analyzed if s.scoring_result is not None and s.scoring_result.recommendation == "Buy")
    watch_count = sum(1 for s in analyzed if s.scoring_result is not None and s.scoring_result.recommendation == "Watch")
    sell_count = sum(1 for s in analyzed if s.scoring_result is not None and s.scoring_result.recommendation == "Sell")

    lines.append(f"Total stocks scanned: {total_all}")
    lines.append(f"Stocks with full analysis: {total_with_data}")
    lines.append(f"Buy signals: {buy_count}, Watch: {watch_count}, Sell: {sell_count}")
    lines.append("")

    # Top stocks with full computed data
    top = [s for s in analyzed if s.scoring_result is not None]
    top.sort(key=lambda x: x.scoring_result.total_score, reverse=True)
    top = top[:20]

    lines.append("TOP RANKED STOCKS (computed data only):")
    lines.append("")
    for s in top:
        sr = s.scoring_result
        lines.append(f"Stock: {s.name} ({s.ticker}) | {s.name_ar}")
        lines.append(f"  Price: {s.current_price} EGP | Change: {s.daily_change_pct}%")
        lines.append(f"  Score: {sr.total_score}/100 | Recommendation: {sr.recommendation}")
        lines.append(f"  Risk: {sr.risk_level} — {sr.risk_reason}")
        lines.append(f"  Data: {sr.data_freshness} (quality: {sr.data_quality:.1%})")
        if s.support > 0:
            lines.append(f"  Support: {s.support} | Resistance: {s.resistance} | R/R: {s.risk_reward_ratio:.1f}")
        lines.append(f"  Key indicators:")
        for ind in s.indicators[:8]:
            lines.append(f"    {ind.name}: {ind.value} ({ind.signal_text}) — {ind.note}")
        if sr.pass_reasons:
            lines.append(f"  Positive factors: {'; '.join(sr.pass_reasons[:3])}")
        if sr.fail_reasons:
            lines.append(f"  Negative factors: {'; '.join(sr.fail_reasons[:3])}")
        lines.append("")

    # Bottom-ranked (bearish)
    bottom = [s for s in analyzed if s.scoring_result is not None]
    bottom.sort(key=lambda x: x.scoring_result.total_score)
    bottom = bottom[:10]

    if bottom:
        lines.append("BOTTOM RANKED STOCKS (bearish):")
        lines.append("")
        for s in bottom:
            sr = s.scoring_result
            if sr.recommendation == "Sell" or sr.total_score < 40:
                lines.append(f"Stock: {s.name} ({s.ticker}) | {s.name_ar}")
                lines.append(f"  Price: {s.current_price} EGP | Score: {sr.total_score}/100 | Rec: {sr.recommendation}")
                if sr.fail_reasons:
                    lines.append(f"  Reasons: {'; '.join(sr.fail_reasons[:3])}")
                lines.append("")

    return "\n".join(lines)


def format_analysis_for_telegram(stocks: list[StockAnalysis]) -> str:
    """Fallback Telegram formatting (no AI)."""
    from ai_report import _escape_markdown
    lines = []
    top_bull = get_top_bullish(stocks, 10)
    top_bear = get_top_bearish(stocks, 5)

    if top_bull:
        lines.append("📊 *أقوى الأسهم صعودًا:*")
        lines.append("")
        for i, s in enumerate(top_bull, 1):
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   {_escape_markdown(str(s.current_price))} | {_escape_markdown(s.signal_label)}")
            if s.bullish_reasons:
                lines.append(f"   {' / '.join(s.bullish_reasons[:3])}")
            lines.append("")

    if top_bear:
        lines.append("⚠️ *أقوى الأسهم هبوطًا:*")
        lines.append("")
        for i, s in enumerate(top_bear, 1):
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   {_escape_markdown(str(s.current_price))} | {_escape_markdown(s.signal_label)}")
            if s.bearish_reasons:
                lines.append(f"   {' / '.join(s.bearish_reasons[:2])}")
            lines.append("")

    return "\n".join(lines)
