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
from data.validator import validate_ohlcv, validate_price, deduplicate_stocks

logger = logging.getLogger(__name__)

STOCKANALYSIS_URL = "https://stockanalysis.com/list/egyptian-stock-exchange/"
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
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        r = requests.get(STOCKANALYSIS_URL, headers=headers, timeout=15)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to scrape stockanalysis.com: {e}")
        return _load_fallback_stock_list()

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        logger.error("No table found on stockanalysis.com page")
        return _load_fallback_stock_list()

    tbody = table.find("tbody")
    if not tbody:
        logger.error("No tbody found in table")
        return _load_fallback_stock_list()

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

            if symbol and symbol != "NO." and validate_price(price):
                raw_stocks.append({
                    "symbol": symbol,
                    "name": name,
                    "price": price,
                    "change_pct": change_pct,
                    "market_cap_str": market_cap_str,
                })

    # Deduplicate
    raw_stocks = deduplicate_stocks(raw_stocks)

    # Update canonical list with freshly scraped symbols
    update_canonical_list(raw_stocks)

    if not raw_stocks:
        logger.warning("Scraping returned 0 valid stocks. Using fallback.")
        return _load_fallback_stock_list()

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
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
            else:
                logger.warning(f"TradingView failed for {ticker} after {retries+1} attempts: {str(e)[:80]}")
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


# ─── Single Stock Scan ───────────────────────────────────────────────────────

def scan_single_stock(ticker: str) -> StockAnalysis | None:
    """
    Scan a single stock by ticker — much faster than scan_all_stocks().
    Used by /stock SYMBOL command.
    """
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

    # Validate
    validation = validate_ohlcv(df, ticker)
    if not validation.is_valid:
        logger.warning(f"Data validation failed for {ticker}: {validation.issues[:2]}")
        return None

    # Analyze
    try:
        analysis = analyze_stock(df, ticker, name_en, name_ar)
        analysis.data_freshness = validation.freshness
        analysis.data_quality = validation.quality_score
        analysis.timestamp = datetime.now().isoformat()
        analysis.scoring_result = compute_score(analysis, validation.freshness, validation.quality_score)
        return analysis
    except Exception as e:
        logger.error(f"Analysis failed for {ticker}: {e}")
        return None



# ─── Full Scan ───────────────────────────────────────────────────────────────

def scan_all_stocks() -> list[StockAnalysis]:
    """
    Full pipeline:
    1. Check cache (return if fresh)
    2. Collect stock list (scrape + validate)
    3. Download historical data per stock (with retries)
    4. Validate OHLCV data
    5. Compute technical indicators
    6. Score each stock (0-100 deterministic)
    7. Cache and return sorted by composite score
    """
    # Check cache first
    cached = _get_cached_scan()
    if cached is not None:
        return cached

    stock_list = scrape_egx_stock_list()
    if not stock_list:
        logger.error("No stocks available. All sources failed.")
        return []

    total = len(stock_list)
    logger.info(f"Starting scan of {total} EGX stocks...")

    results: list[StockAnalysis] = []
    success_count = 0
    fail_count = 0
    rejected_count = 0

    for i, stock_info in enumerate(stock_list, 1):
        ticker = stock_info["symbol"]
        name_en = stock_info["name"]
        live_price = stock_info["price"]
        live_change = stock_info["change_pct"]

        # Validate symbol
        if not is_valid_egx_symbol(ticker):
            logger.debug(f"  ❌ {ticker}: not a verified EGX symbol, skipping")
            rejected_count += 1
            continue

        name_ar = get_arabic_name(ticker, name_en)

        if i % 20 == 0:
            logger.info(f"Progress: {i}/{total} ({success_count} ok, {fail_count} failed, {rejected_count} rejected)")

        # Download historical data
        df = download_stock_history(ticker, n_bars=250, retries=2)

        # Validate OHLCV data
        freshness = "unknown"
        data_quality = 0.5

        if df is not None and len(df) >= 50:
            validation = validate_ohlcv(df, ticker)
            freshness = validation.freshness
            data_quality = validation.quality_score

            if not validation.is_valid:
                logger.debug(f"  ⚠️ {ticker}: data validation failed — {validation.issues[0] if validation.issues else 'unknown'}")
                fail_count += 1
                if live_price > 0:
                    results.append(StockAnalysis(
                        ticker=ticker, name=name_en, name_ar=name_ar,
                        current_price=live_price, daily_change_pct=live_change, volume=0,
                        data_freshness=freshness, data_quality=data_quality,
                    ))
                continue
        elif df is None or len(df) < 50:
            fail_count += 1
            if live_price > 0:
                results.append(StockAnalysis(
                    ticker=ticker, name=name_en, name_ar=name_ar,
                    current_price=live_price, daily_change_pct=live_change, volume=0,
                    data_freshness="unknown", data_quality=0.3,
                ))
            continue

        # Compute indicators
        try:
            analysis = analyze_stock(df, ticker, name_en, name_ar)

            # Override with live price from stockanalysis.com
            if live_price > 0:
                analysis.current_price = live_price
                analysis.daily_change_pct = live_change

            # Attach data quality metadata
            analysis.data_freshness = freshness
            analysis.data_quality = data_quality
            analysis.timestamp = datetime.now().isoformat()

            # Compute deterministic score
            scoring = compute_score(analysis, freshness, data_quality)
            analysis.scoring_result = scoring  # type: ignore[attr-defined]

            results.append(analysis)
            success_count += 1

        except Exception as e:
            logger.warning(f"  ❌ {ticker}: analysis failed: {str(e)[:80]}")
            fail_count += 1
            if live_price > 0:
                results.append(StockAnalysis(
                    ticker=ticker, name=name_en, name_ar=name_ar,
                    current_price=live_price, daily_change_pct=live_change, volume=0,
                    data_freshness="unknown", data_quality=0.3,
                ))

        time.sleep(0.15)

    results.sort(key=lambda x: x.composite_score, reverse=True)
    _set_cached_scan(results)
    logger.info(f"Scan complete: {success_count} analyzed, {fail_count} failed, {rejected_count} rejected, {total} total")
    return results


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
