"""
stock_scanner.py
----------------
Scans all available EGX stocks on Yahoo Finance, calculates technical
indicators for each, and ranks them by bullish signal strength.

Uses yfinance (free) for historical OHLCV data.
Yahoo Finance uses the .CA suffix for Cairo Stock Exchange.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from indicators import StockAnalysis, analyze_stock

logger = logging.getLogger(__name__)

# ─── EGX Stock Universe ──────────────────────────────────────────────────────

# Major EGX stocks available on Yahoo Finance (.CA = Cairo)
# Format: (ticker, name_en, name_ar)
EGX_STOCKS: list[tuple[str, str, str]] = [
    ("COMI.CA",  "Commercial International Bank", "البنك التجاري الدولي"),
    ("SWDY.CA",  "Ezz Steel",                      "عز الدخاخني"),
    ("HRHO.CA",  "EFG Hermes",                     "إي إف جي هيرميس"),
    ("ETEL.CA",  "Telecom Egypt",                  "المصرية للاتصالات"),
    ("JUFO.CA",  "Juhayna Food",                   "جهينة للصناعات الغذائية"),
    ("EFIH.CA",  "eFinance",                       "إي فاينانس"),
    ("EFID.CA",  "Edita Food",                     "إديتا"),
    ("ORAS.CA",  "Orascom Construction",           "أوراسكوم للإنشاء"),
    ("ORWE.CA",  "Orascom Development",            "أوراسكوم للتنمية"),
    ("CCAP.CA",  "Credit Agricole Egypt",          "كريدي أجريكول مصر"),
    ("FWRY.CA",  "Fawry",                          "فوري"),
    ("PHDC.CA",  "Palm Hills",                     "بالم هيلز"),
    ("CIRA.CA",  "Cira Education",                 "سيرا للتعليم"),
    ("RAYA.CA",  "Raya Holding",                   "رايا القابضة"),
    ("EKHO.CA",  "Egyptian Kuwaiti Holding",       "المصرية الكويتية القابضة"),
    ("AMOC.CA",  "Alexandria Mineral Oils",        "العامة للبترول"),
    ("ABUK.CA",  "Abu Kir Fertilizers",            "أبو قير للأسمدة"),
    ("MFPC.CA",  "Misr Fertilizers",               "مصر للأسمدة"),
    ("EAST.CA",  "Eastern Company",                "الشرقية للدخان"),
    ("SKPC.CA",  "Sidi Kerir Petrochemicals",      "سيدي كرير للبتروكيماويات"),
]

# Download period — need 200+ days for SMA200
DOWNLOAD_PERIOD = "1y"


def fetch_stock_data(ticker: str) -> Optional[pd.DataFrame]:
    """Download historical OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(ticker, period=DOWNLOAD_PERIOD, progress=False, auto_adjust=True)
        if df is not None and len(df) > 0:
            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        return None
    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def scan_all_stocks() -> list[StockAnalysis]:
    """
    Scan all EGX stocks, calculate indicators, and return sorted analysis list.
    Sorted by composite score (most bullish first).
    """
    results: list[StockAnalysis] = []
    total = len(EGX_STOCKS)

    for i, (ticker, name_en, name_ar) in enumerate(EGX_STOCKS, 1):
        logger.info(f"[{i}/{total}] Analyzing {ticker} ({name_en})...")

        df = fetch_stock_data(ticker)

        if df is None or len(df) < 50:
            logger.warning(f"  ⚠️ Insufficient data for {ticker}, skipping.")
            continue

        try:
            analysis = analyze_stock(df, ticker, name_en, name_ar)
            results.append(analysis)
            logger.info(
                f"  📊 {ticker}: score={analysis.composite_score} "
                f"label={analysis.signal_label} price={analysis.current_price}"
            )
        except Exception as e:
            logger.error(f"  ❌ Analysis failed for {ticker}: {e}")

        # Small delay to avoid rate limiting on Yahoo Finance
        time.sleep(0.3)

    # Sort by composite score (most bullish first)
    results.sort(key=lambda x: x.composite_score, reverse=True)
    return results


def get_top_bullish(stocks: list[StockAnalysis], top_n: int = 5) -> list[StockAnalysis]:
    """Return top N stocks with the strongest bullish signals."""
    bullish = [s for s in stocks if s.composite_score >= 2]
    return bullish[:top_n]


def get_top_bearish(stocks: list[StockAnalysis], top_n: int = 3) -> list[StockAnalysis]:
    """Return top N stocks with the strongest bearish signals."""
    bearish = [s for s in stocks if s.composite_score <= -2]
    # Sort most bearish first
    bearish.sort(key=lambda x: x.composite_score)
    return bearish[:top_n]


def format_analysis_for_ai(stocks: list[StockAnalysis]) -> str:
    """
    Format stock analysis data as text for Gemini AI to generate Arabic report.
    Includes all scanned stocks with their indicator details.
    """
    lines = ["EGX STOCK TECHNICAL ANALYSIS REPORT", "=" * 50, ""]

    for s in stocks:
        lines.append(f"Stock: {s.name} ({s.ticker})")
        lines.append(f"  Arabic Name: {s.name_ar}")
        lines.append(f"  Price: {s.current_price}")
        lines.append(f"  Daily Change: {s.daily_change_pct}%")
        lines.append(f"  Volume: {s.volume:,}")
        lines.append(f"  Signal: {s.signal_label} (score: {s.composite_score})")
        lines.append(f"  Confidence: {s.signal_score_pct}%")

        lines.append("  Indicators:")
        for ind in s.indicators:
            lines.append(f"    {ind.name}: {ind.value} ({ind.signal_text}) — {ind.note}")

        if s.volume_profile:
            vp = s.volume_profile
            lines.append(f"    Volume Profile: POC={vp.poc}, VA=[{vp.value_area_low}-{vp.value_area_high}]")
            lines.append(f"    Price position: {vp.current_price_position} ({vp.signal_text})")

        if s.bullish_reasons:
            lines.append(f"  Bullish reasons: {'; '.join(s.bullish_reasons)}")
        if s.bearish_reasons:
            lines.append(f"  Bearish reasons: {'; '.join(s.bearish_reasons)}")

        lines.append("")

    # Summary section
    top_bull = get_top_bullish(stocks, 5)
    top_bear = get_top_bearish(stocks, 3)

    lines.append("TOP BULLISH CANDIDATES:")
    for s in top_bull:
        lines.append(f"  {s.name_ar} ({s.ticker}): {s.signal_label} — score {s.composite_score}")
    lines.append("")

    lines.append("TOP BEARISH CANDIDATES:")
    for s in top_bear:
        lines.append(f"  {s.name_ar} ({s.ticker}): {s.signal_label} — score {s.composite_score}")

    return "\n".join(lines)


def format_analysis_for_telegram(stocks: list[StockAnalysis]) -> str:
    """
    Format stock analysis as a human-readable Telegram message section.
    Used as fallback if Gemini AI is unavailable.
    """
    lines = []
    top_bull = get_top_bullish(stocks, 5)
    top_bear = get_top_bearish(stocks, 3)

    if top_bull:
        lines.append("📊 *أقوى الأسهم صعودًا (بناءً على المؤشرات التقنية):*")
        lines.append("")
        for i, s in enumerate(top_bull, 1):
            lines.append(f"{i}\\. *{s.name_ar}* ({s.ticker.replace('.CA', '')})")
            lines.append(f"   السعر: {s.current_price} | التغيير: {s.daily_change_pct}%")
            lines.append(f"   الإشارة: {s.signal_label} | الثقة: {s.signal_score_pct}%")
            if s.bullish_reasons:
                lines.append(f"   الأسباب: {' / '.join(s.bullish_reasons[:3])}")
            lines.append("")

    if top_bear:
        lines.append("⚠️ *أقوى الأسهم هبوطًا:*")
        lines.append("")
        for i, s in enumerate(top_bear, 1):
            lines.append(f"{i}\\. *{s.name_ar}* ({s.ticker.replace('.CA', '')})")
            lines.append(f"   السعر: {s.current_price} | الإشارة: {s.signal_label}")
            if s.bearish_reasons:
                lines.append(f"   الأسباب: {' / '.join(s.bearish_reasons[:2])}")
            lines.append("")

    return "\n".join(lines)
