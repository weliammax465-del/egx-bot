"""
stock_scanner.py
----------------
Scans ALL EGX stocks (224+) with real-time prices and professional technical analysis.

Data sources:
  1. stockanalysis.com — scraped for the full list of all EGX stocks with 
     current prices and daily changes (real-time data)
  2. TradingView (tvdatafeed) — historical OHLCV data for technical indicators

The scanner:
  - Scrapes the full EGX stock list with live prices
  - Downloads 250 days of historical data per stock from TradingView
  - Calculates 9+ technical indicators per stock
  - Ranks stocks by composite bullish/bearish score
  - Returns the top candidates for the AI report
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from indicators import StockAnalysis, analyze_stock

logger = logging.getLogger(__name__)

# ─── Arabic Names for Major EGX Stocks ───────────────────────────────────────

ARABIC_NAMES = {
    "COMI": "البنك التجاري الدولي", "TMGH": "طلعت مصطفى", "SWDY": "السويدي إليكتريك",
    "ETEL": "المصرية للاتصالات", "EGAL": "مصر للألمنيوم", "EAST": "الشرقية للدخان",
    "QNBE": "بنك قطر الوطني", "MFPC": "مصر للأسمدة", "ABUK": "أبو قير للأسمدة",
    "HDBK": "بنك الإسكان والتعمير", "ALCN": "الإسكندرية للحاويات", "ORAS": "أوراسكوم للإنشاء",
    "EFIH": "إي فاينانس", "ADIB": "بنك أبوظبي الإسلامي", "EMFD": "عمار مصر",
    "FWRY": "فوري", "SCTS": "قناة السويس للتكنولوجيا", "ORHD": "أوراسكوم للتنمية",
    "PHDC": "بالم هيلز", "GPPL": "الأهرام القابضة", "VLMR": "فالمور",
    "HRHO": "إي إف جي هيرميس", "EFID": "إديتا", "JUFO": "جهينة",
    "CANA": "بنك قناة السويس", "GBCO": "جي بي كورب", "OCDI": "سوديك",
    "BTFH": "بلتون", "RAYA": "رايا القابضة", "IRON": "ال الحديد والصلب",
    "FERC": "فيركيم", "CIEB": "كريدي أجريكول", "FAIT": "بنك فيصل الإسلامي",
    "HELI": "مدينة هليوبوليس", "EGCH": "الكيماويات المصرية", "VALU": "فاليو",
    "EXPA": "بنك التنمية الصادرات", "CLHO": "مستشفيات كليوباترا", "ARCC": "أسمنت العربية",
    "CCAP": "قالا", "TAQA": "طاقة", "EFIC": "الصناعات المالية والصناعية",
    "POUL": "دواجن القاهرة", "SKPC": "سيدي كرير", "EGTS": "المنتجعات المصرية",
    "MTIE": "مجموعة أم أم", "CIRA": "سيرا للتعليم", "SCEM": "أسمنت سيناء",
    "EGSA": "نايل سات", "MCQE": "أسمنت قنا", "SAUD": "بنك البركة",
    "ORWE": "نسج الشرق", "MASR": "مدينة مصر", "PHAR": "إيبيكو",
    "UBEE": "البنك المتحد", "MHOT": "فنادق مصر", "MBSC": "أسمنت بني سويف",
    "ISPH": "ابن سينا فارما", "CICH": "سي آي كابيتال", "EGBE": "بنك الخليج",
    "TALM": "تعليم", "ATQA": "عتاقة للصلب", "MOIL": "ماريدايف",
    "BINV": "استثمارات ب", "RMDA": "راميدا", "AMOC": "العامة للبترول",
    "IFAP": "المحاصيل الزراعية", "CSAG": "وكالات الشحن بالقناة", "OLFI": "أوبور لاند",
    "ISMQ": "مناجم الحديد", "BONY": "بنيان", "NIPH": "نايل فارما",
    "DOMT": "دومتي", "MIPH": "مينافارم", "KORA": "كورة للطاقة",
    "OIH": "أوراسكوم للاستثمار", "PRDC": "رواد العقارية", "MPRC": "مدينة الإنتاج",
    "EGAS": "غاز مصر", "ELEC": "كابلات مصر", "SUGR": "سكر الدلتا",
    "ZMID": "الزهراء المعادي", "ACAP": "أيه كابيتال", "AMES": "مركز الإسكندرية الطبي",
    "MOIN": "مهندس للتأمين", "BIOC": "جلاكسو", "PHTV": "بيراميزا",
    "NAPR": "الطباعة الوطنية", "CNFN": "كونتاكت", "CPCI": "كحيرة للأدوية",
    "AXPH": "الإسكندرية للأدوية", "NINH": "مستشفى النزهة", "MPCI": "ممفيس للأدوية",
    "ENGC": "آيكون", "GOUR": "جورميه", "SPIN": "الغزل والنسيج",
    "DSCW": "دايس", "MFSC": "محلات مصر الحرة", "SVCE": "أسمنت وادي النيل",
    "AMIA": "عرب ملتقى", "GSSC": "الصوامع العامة", "GDWA": "جدوة",
    "OCPH": "أكتوبر فارما", "MICH": "الكيماويات المصرية صناعات", "WCDF": "مطاحن الدلتا",
    "AJWA": "أجوة", "KABO": "كابو", "SAIB": "البنك العربي الأفريقي",
    "UEFM": "مطاحن صعيد مصر", "ACTF": "أكت فايننشال", "UNIT": "المتحدة للإسكان",
    "ASCM": "أسكوم", "ADCI": "الدواء العربي", "ARAB": "العربية للتعمير",
    "OFH": "أو بي", "ACAMD": "إدارة الأصول العربية", "ISMA": "إسماعيلية للدواجن",
    "ELSH": "الشمس للإسكان", "ETRS": "إيجي ترانس", "SDTI": "شرم دريمز",
    "KZPC": "كفر الزيات", "ACGC": "القطن العربية", "LCSW": "ليسيكو",
    "CFGH": "كونكريت فاشون", "ALRA": "أطلس", "ELKA": "القاهرة للإسكان",
    "AFMC": "مطاحن الإسكندرية", "ZEOT": "الزيوت", "AMER": "أمير جروب",
    "ATLC": "التوفيق للتأجير", "PHGC": "بريميوم هيلث", "SNFC": "أمن الغذاء بالشرقية",
    "EDFM": "مطاحن الدلتا الشرقية", "NAHO": "نعيم", "GGRN": "جو جرين",
    "DAPH": "دي إف جي", "INFI": "إسماعيلية للصناعات الغذائية",
}


# ─── Stock List Scraping ─────────────────────────────────────────────────────

STOCKANALYSIS_URL = "https://stockanalysis.com/list/egyptian-stock-exchange/"


def scrape_egx_stock_list() -> list[dict]:
    """
    Scrape the full list of all EGX stocks from stockanalysis.com.
    Returns list of dicts with: symbol, name, price, change_pct, market_cap_str.
    This gives us real-time current prices for all 224 stocks.
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
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        logger.error("No table found on stockanalysis.com page")
        return []

    tbody = table.find("tbody")
    if not tbody:
        logger.error("No tbody found in table")
        return []

    stocks = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) >= 5:
            symbol = cells[1].get_text(strip=True)
            name = cells[2].get_text(strip=True)
            price_str = cells[4].get_text(strip=True)
            change_str = cells[5].get_text(strip=True)
            market_cap_str = cells[3].get_text(strip=True) if len(cells) > 3 else ""

            # Parse price
            try:
                price = float(price_str.replace(",", ""))
            except (ValueError, TypeError):
                price = 0.0

            # Parse change %
            change_pct = 0.0
            try:
                change_clean = change_str.replace("%", "").replace("+", "").strip()
                change_pct = float(change_clean)
                if change_str.strip().startswith("-"):
                    change_pct = -change_pct
            except (ValueError, TypeError):
                pass

            if symbol and symbol != "No.":
                stocks.append({
                    "symbol": symbol.replace(".CA", ""),  # Clean ticker
                    "name": name,
                    "price": price,
                    "change_pct": change_pct,
                    "market_cap_str": market_cap_str,
                })

    logger.info(f"Scraped {len(stocks)} EGX stocks from stockanalysis.com")
    return stocks


# ─── TradingView Historical Data ─────────────────────────────────────────────

_tv_instance = None


def _get_tv():
    """Lazy-initialize TradingView datafeed instance."""
    global _tv_instance
    if _tv_instance is None:
        from tvDatafeed import TvDatafeed, Interval
        _tv_instance = TvDatafeed()
    return _tv_instance


def download_stock_history(ticker: str, n_bars: int = 250) -> Optional[pd.DataFrame]:
    """
    Download historical OHLCV data from TradingView.
    Returns DataFrame with capitalized column names (Open, High, Low, Close, Volume).
    """
    from tvDatafeed import TvDatafeed, Interval

    tv = _get_tv()
    try:
        df = tv.get_hist(symbol=ticker, exchange="EGX", interval=Interval.in_daily, n_bars=n_bars)
        if df is None or len(df) == 0:
            return None

        # Rename columns to match indicators.py expectations
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        # Drop the symbol column if present
        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        return df
    except Exception as e:
        logger.warning(f"TradingView download failed for {ticker}: {str(e)[:80]}")
        return None


# ─── Full Scan ───────────────────────────────────────────────────────────────

def scan_all_stocks() -> list[StockAnalysis]:
    """
    Full scan of all EGX stocks:
    1. Scrape stockanalysis.com for all 224 stocks with real-time prices
    2. Download historical data from TradingView for each
    3. Calculate technical indicators
    4. Return sorted by composite score (most bullish first)
    """
    # Step 1: Get the stock list with live prices
    stock_list = scrape_egx_stock_list()

    if not stock_list:
        logger.error("No stocks scraped. Falling back to empty list.")
        return []

    total = len(stock_list)
    logger.info(f"Starting scan of {total} EGX stocks...")

    results: list[StockAnalysis] = []
    success_count = 0
    fail_count = 0

    for i, stock_info in enumerate(stock_list, 1):
        ticker = stock_info["symbol"]
        name_en = stock_info["name"]
        name_ar = ARABIC_NAMES.get(ticker, name_en)
        live_price = stock_info["price"]
        live_change = stock_info["change_pct"]

        if i % 20 == 0:
            logger.info(f"Progress: {i}/{total} ({success_count} ok, {fail_count} failed)")

        # Step 2: Download historical data from TradingView
        df = download_stock_history(ticker, n_bars=250)

        if df is None or len(df) < 50:
            logger.debug(f"  ⚠️ {ticker}: insufficient data ({len(df) if df is not None else 0} bars), skipping indicators")
            fail_count += 1
            # Still include with live price but no indicators
            if live_price > 0:
                analysis = StockAnalysis(
                    ticker=ticker,
                    name=name_en,
                    name_ar=name_ar,
                    current_price=live_price,
                    daily_change_pct=live_change,
                    volume=0,
                    signal_label="لا توجد بيانات كافية ⚪",
                )
                results.append(analysis)
            continue

        # Step 3: Calculate indicators
        try:
            analysis = analyze_stock(df, ticker, name_en, name_ar)

            # Override with live price from stockanalysis.com (more real-time)
            if live_price > 0:
                analysis.current_price = live_price
                analysis.daily_change_pct = live_change

            results.append(analysis)
            success_count += 1

            logger.debug(f"  📊 {ticker}: score={analysis.composite_score} {analysis.signal_label}")

        except Exception as e:
            logger.warning(f"  ❌ {ticker}: analysis failed: {str(e)[:80]}")
            fail_count += 1

        # Small delay to avoid rate limiting
        time.sleep(0.15)

    # Sort by composite score (most bullish first)
    results.sort(key=lambda x: x.composite_score, reverse=True)

    logger.info(f"Scan complete: {success_count} analyzed, {fail_count} skipped, {total} total")
    return results


def get_top_bullish(stocks: list[StockAnalysis], top_n: int = 10) -> list[StockAnalysis]:
    """Return top N stocks with the strongest bullish signals."""
    bullish = [s for s in stocks if s.composite_score >= 2]
    return bullish[:top_n]


def get_top_bearish(stocks: list[StockAnalysis], top_n: int = 5) -> list[StockAnalysis]:
    """Return top N stocks with the strongest bearish signals."""
    bearish = [s for s in stocks if s.composite_score <= -2]
    bearish.sort(key=lambda x: x.composite_score)
    return bearish[:top_n]


def format_analysis_for_ai(stocks: list[StockAnalysis]) -> str:
    """
    Format stock analysis data as text for Gemini AI.
    To stay within free-tier token limits, only sends:
    - Top 20 bullish stocks with full indicator details
    - Top 10 bearish stocks with full indicator details
    - Market-wide statistics summary
    """
    lines = ["EGX STOCK TECHNICAL ANALYSIS REPORT", "=" * 50, ""]

    # Only include stocks that actually have indicators
    analyzed = [s for s in stocks if s.indicators]
    total_with_data = len(analyzed)
    total_all = len(stocks)

    # Market stats
    bullish_count = sum(1 for s in analyzed if s.composite_score >= 2)
    bearish_count = sum(1 for s in analyzed if s.composite_score <= -2)
    neutral_count = total_with_data - bullish_count - bearish_count

    lines.append(f"Total stocks scanned: {total_all}")
    lines.append(f"Stocks with full analysis: {total_with_data}")
    lines.append(f"Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}")
    lines.append("")

    # Top bullish — full details (limit to 20 to control token usage)
    top_bull = get_top_bullish(stocks, 20)
    lines.append("TOP BULLISH CANDIDATES (detailed):")
    lines.append("")
    for s in top_bull:
        lines.append(f"Stock: {s.name} ({s.ticker}) | {s.name_ar}")
        lines.append(f"  Price: {s.current_price} | Change: {s.daily_change_pct}%")
        lines.append(f"  Signal: {s.signal_label} (score: {s.composite_score}, confidence: {s.signal_score_pct}%)")
        lines.append("  Indicators:")
        for ind in s.indicators:
            lines.append(f"    {ind.name}: {ind.value} ({ind.signal_text}) — {ind.note}")
        if s.volume_profile:
            vp = s.volume_profile
            lines.append(f"    Volume Profile: POC={vp.poc}, VA=[{vp.value_area_low}-{vp.value_area_high}], Position: {vp.current_price_position}")
        if s.bullish_reasons:
            lines.append(f"  Bullish reasons: {'; '.join(s.bullish_reasons)}")
        lines.append("")

    # Top bearish — full details
    top_bear = get_top_bearish(stocks, 10)
    lines.append("TOP BEARISH CANDIDATES (detailed):")
    lines.append("")
    for s in top_bear:
        lines.append(f"Stock: {s.name} ({s.ticker}) | {s.name_ar}")
        lines.append(f"  Price: {s.current_price} | Change: {s.daily_change_pct}%")
        lines.append(f"  Signal: {s.signal_label} (score: {s.composite_score})")
        lines.append("  Indicators:")
        for ind in s.indicators:
            lines.append(f"    {ind.name}: {ind.value} ({ind.signal_text}) — {ind.note}")
        if s.bearish_reasons:
            lines.append(f"  Bearish reasons: {'; '.join(s.bearish_reasons)}")
        lines.append("")

    return "\n".join(lines)


def format_analysis_for_telegram(stocks: list[StockAnalysis]) -> str:
    """
    Format stock analysis as a human-readable Telegram message section.
    Used as fallback if Gemini AI is unavailable.
    """
    from ai_report import _escape_markdown

    lines = []
    top_bull = get_top_bullish(stocks, 10)
    top_bear = get_top_bearish(stocks, 5)

    if top_bull:
        lines.append("📊 *أقوى الأسهم صعودًا (بناءً على المؤشرات التقنية):*")
        lines.append("")
        for i, s in enumerate(top_bull, 1):
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   السعر: {_escape_markdown(str(s.current_price))} | التغيير: {_escape_markdown(str(s.daily_change_pct))}%")
            lines.append(f"   الإشارة: {_escape_markdown(s.signal_label)} | الثقة: {_escape_markdown(str(s.signal_score_pct))}%")
            if s.bullish_reasons:
                lines.append(f"   الأسباب: {' / '.join(s.bullish_reasons[:3])}")
            lines.append("")

    if top_bear:
        lines.append("⚠️ *أقوى الأسهم هبوطًا:*")
        lines.append("")
        for i, s in enumerate(top_bear, 1):
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   السعر: {_escape_markdown(str(s.current_price))} | الإشارة: {_escape_markdown(s.signal_label)}")
            if s.bearish_reasons:
                lines.append(f"   الأسباب: {' / '.join(s.bearish_reasons[:2])}")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick local test
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=== Full EGX Stock Scan ===")
    stocks = scan_all_stocks()
    print(f"\nAnalyzed {len(stocks)} stocks\n")

    print("=== Top 10 Bullish ===")
    for s in get_top_bullish(stocks, 10):
        print(f"  🟢 {s.name_ar}: {s.signal_label} (score={s.composite_score})")

    print("\n=== Top 5 Bearish ===")
    for s in get_top_bearish(stocks, 5):
        print(f"  🔴 {s.name_ar}: {s.signal_label} (score={s.composite_score})")
