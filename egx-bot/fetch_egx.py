"""
fetch_egx.py
------------
Fetches Egyptian Exchange (EGX) market data from public sources.

Primary source: Trading Economics (tradingeconomics.com)
Provides: EGX 30 index value, daily change, monthly & yearly performance.

No API keys required. No sensitive data stored.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional
import logging

from data.validator import classify_scrape_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TRADING_ECONOMICS_URL = "https://tradingeconomics.com/egypt/stock-market"


@dataclass
class MarketSummary:
    index_name: str
    current_value: Optional[str]
    change: Optional[str]
    change_pct: Optional[str]
    direction: str  # "up", "down", "flat"
    month_change_pct: Optional[str] = None
    year_change_pct: Optional[str] = None
    date_str: Optional[str] = None
    top_gainers: list = field(default_factory=list)
    top_losers: list = field(default_factory=list)
    most_active: list = field(default_factory=list)
    source_note: str = ""
    is_trading_day: bool = True


def _safe_get(url: str, timeout: int = 15, retries: int = 2) -> Optional[requests.Response]:
    """HTTP GET with retry logic."""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt  # 1s, 2s
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{retries + 1} failed for {url}: {e}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
            else:
                logger.warning(f"Failed to fetch {url} after {retries + 1} attempts: {e}")
                return None


def _parse_number(text: str) -> Optional[float]:
    """Extract a float from a string like '51,443.07' or '-267.83'."""
    if not text:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def fetch_egx30_index() -> dict:
    """
    Scrape EGX 30 index data from Trading Economics.
    Returns dict with: value, change, change_pct, month_change_pct,
    year_change_pct, direction, date_str.
    """
    resp = _safe_get(TRADING_ECONOMICS_URL)
    if not resp:
        logger.error("Trading Economics: no response after retries — classified as 'no_response'")
        return {"_error": "no_response", "_source": "tradingeconomics.com"}

    soup = BeautifulSoup(resp.text, "html.parser")
    data = {}

    try:
        # Trading Economics has the EGX 30 in a table on the Egypt stock market page
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2 and cells[0].get_text(strip=True) == "EGX 30":
                    # Parse the row: [name, price, _, change, day%, month%, year%, date]
                    if len(cells) >= 7:
                        data["value"] = cells[1].get_text(strip=True)
                        data["change"] = cells[3].get_text(strip=True)
                        data["change_pct"] = cells[4].get_text(strip=True)
                        data["month_change_pct"] = cells[5].get_text(strip=True)
                        data["year_change_pct"] = cells[6].get_text(strip=True)
                    elif len(cells) >= 5:
                        data["value"] = cells[1].get_text(strip=True)
                        data["change"] = cells[3].get_text(strip=True)
                        data["change_pct"] = cells[4].get_text(strip=True)

                    # Date might be in the last cell
                    if len(cells) >= 8:
                        data["date_str"] = cells[7].get_text(strip=True)

                    # Validate that we got actual numbers, not empty strings
                    parsed_value = _parse_number(data.get("value", ""))
                    if parsed_value is None or parsed_value <= 0:
                        logger.warning(f"Trading Economics: EGX 30 value is invalid: '{data.get('value', '')}'")
                        return {"_error": "invalid_data", "_source": "tradingeconomics.com"}

                    # Determine direction from change value
                    change_val = _parse_number(data.get("change", "0"))
                    if change_val is not None:
                        if change_val < 0:
                            data["direction"] = "down"
                        elif change_val > 0:
                            data["direction"] = "up"
                        else:
                            data["direction"] = "flat"
                    else:
                        data["direction"] = "flat"

                    logger.info(
                        f"EGX 30: {data.get('value', 'N/A')} "
                        f"({data.get('change', 'N/A')}, {data.get('change_pct', 'N/A')})"
                    )
                    return data

        # If we got here, the page loaded but EGX 30 row wasn't found
        logger.warning("Trading Economics: page structure changed — EGX 30 row not found in tables")
        return {"_error": "page_structure_changed", "_source": "tradingeconomics.com"}
    except Exception as e:
        error_type = classify_scrape_error(e, "tradingeconomics.com")
        logger.warning(f"Trading Economics: error parsing page — type: {error_type} | {e}")
        return {"_error": error_type, "_source": "tradingeconomics.com"}


def build_market_summary() -> MarketSummary:
    """
    Aggregate all market data into a single MarketSummary object.
    Falls back gracefully — partial data is better than no data.
    """
    logger.info("Fetching EGX 30 index data from Trading Economics…")
    index_data = fetch_egx30_index()

    value = index_data.get("value", "N/A")
    change = index_data.get("change", "N/A")
    change_pct = index_data.get("change_pct", "N/A")
    direction = index_data.get("direction", "flat")
    month_pct = index_data.get("month_change_pct")
    year_pct = index_data.get("year_change_pct")
    date_str = index_data.get("date_str")

    has_data = value != "N/A"

    return MarketSummary(
        index_name="EGX 30",
        current_value=value,
        change=change,
        change_pct=change_pct,
        direction=direction,
        month_change_pct=month_pct,
        year_change_pct=year_pct,
        date_str=date_str,
        source_note="البيانات من Trading Economics." if has_data
                    else "تعذّر جلب البيانات. قد تكون البورصة مغلقة.",
        is_trading_day=has_data,
    )


def format_summary_text(summary: MarketSummary) -> str:
    """
    Format market summary as plain text for passing to the AI summarizer.
    """
    arrow = "📈" if summary.direction == "up" else ("📉" if summary.direction == "down" else "➡️")

    lines = [
        f"EGX 30 Index: {summary.current_value} {arrow}",
        f"Change: {summary.change} ({summary.change_pct})",
    ]

    if summary.month_change_pct:
        lines.append(f"Monthly Change: {summary.month_change_pct}")
    if summary.year_change_pct:
        lines.append(f"Yearly Change: {summary.year_change_pct}")

    return "\n".join(lines)


if __name__ == "__main__":
    summary = build_market_summary()
    print(format_summary_text(summary))
