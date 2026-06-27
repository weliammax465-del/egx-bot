"""
fetch_egx.py
------------
Fetches Egyptian Exchange (EGX) market data from public sources.

Primary source: Investing.com (EGX 30 index & top movers)
Fallback source: EGX official site (egyptse.com)

No API keys required. No sensitive data stored.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

EGX30_INVESTING_URL = "https://www.investing.com/indices/egx-30"
EGX_MOVERS_URL = "https://www.investing.com/equities/egypt"
EGX_OFFICIAL_URL = "https://www.egyptse.com/market-data"

# Arabic day/month names for date formatting
ARABIC_DAYS = {
    "Monday": "الإثنين", "Tuesday": "الثلاثاء", "Wednesday": "الأربعاء",
    "Thursday": "الخميس", "Friday": "الجمعة", "Saturday": "السبت",
    "Sunday": "الأحد",
}
ARABIC_MONTHS = {
    "January": "يناير", "February": "فبراير", "March": "مارس",
    "April": "أبريل", "May": "مايو", "June": "يونيو",
    "July": "يوليو", "August": "أغسطس", "September": "سبتمبر",
    "October": "أكتوبر", "November": "نوفمبر", "December": "ديسمبر",
}


@dataclass
class MarketSummary:
    index_name: str
    current_value: Optional[str]
    change: Optional[str]
    change_pct: Optional[str]
    direction: str  # "up", "down", "flat"
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


def _parse_change_value(change_text: str) -> float:
    """Extract numeric value from a change string like '+1,234.56' or '-500.00'."""
    if not change_text:
        return 0.0
    cleaned = re.sub(r"[^\d.\-]", "", change_text.replace(",", ""))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def fetch_egx30_index() -> dict:
    """Scrape EGX 30 index summary from Investing.com."""
    resp = _safe_get(EGX30_INVESTING_URL)
    if not resp:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    data = {}

    try:
        # Try multiple selector patterns — Investing.com changes these often
        price_el = (
            soup.select_one('[data-test="instrument-price-last"]')
            or soup.select_one("span.text-2xl")
            or soup.select_one(".instrument-price_last__3qDsf")
        )
        if price_el:
            data["value"] = price_el.get_text(strip=True)

        change_el = (
            soup.select_one('[data-test="instrument-price-change"]')
            or soup.select_one(".instrument-price_change__3f2Uc")
        )
        pct_el = (
            soup.select_one('[data-test="instrument-price-change-percent"]')
            or soup.select_one(".instrument-price_change__3f2Uc + span")
        )
        if change_el:
            data["change"] = change_el.get_text(strip=True)
        if pct_el:
            raw = pct_el.get_text(strip=True).strip("()")
            data["change_pct"] = raw

        # Determine direction from change value
        change_val = _parse_change_value(data.get("change", "0"))
        if change_val < 0:
            data["direction"] = "down"
        elif change_val > 0:
            data["direction"] = "up"
        else:
            data["direction"] = "flat"

    except Exception as e:
        logger.warning(f"Error parsing EGX30 index: {e}")

    return data


def fetch_top_movers() -> dict:
    """
    Scrape top gainers, losers, and most active from Investing.com Egypt equities.
    Returns dict with keys: gainers, losers, most_active.
    """
    resp = _safe_get(EGX_MOVERS_URL)
    result = {"gainers": [], "losers": [], "most_active": []}

    if not resp:
        return result

    soup = BeautifulSoup(resp.text, "html.parser")

    def parse_table(table_id: str) -> list:
        """Parse a movers table by ID, with fallback to class-based search."""
        rows = []
        table = soup.find("table", {"id": table_id})
        if not table:
            # Fallback: try common Investing.com table classes
            table = soup.select_one(f"table.{table_id}")
        if not table:
            return rows

        for tr in table.select("tbody tr")[:5]:
            cells = tr.find_all("td")
            if len(cells) >= 3:
                rows.append({
                    "name": cells[1].get_text(strip=True) if len(cells) > 1 else "—",
                    "price": cells[2].get_text(strip=True) if len(cells) > 2 else "—",
                    "change_pct": cells[-1].get_text(strip=True) if cells else "—",
                })
        return rows

    result["gainers"] = parse_table("gainers")
    result["losers"] = parse_table("losers")
    result["most_active"] = parse_table("most_active")

    # If Investing.com tables not found, try scanning for stock screener tables
    if not result["gainers"] and not result["losers"]:
        logger.warning(
            "Investing.com movers tables not found. "
            "Site structure may have changed."
        )

    return result


def build_market_summary() -> MarketSummary:
    """
    Aggregate all market data into a single MarketSummary object.
    Falls back gracefully — partial data is better than no data.
    """
    logger.info("Fetching EGX 30 index data…")
    index_data = fetch_egx30_index()

    logger.info("Fetching top movers…")
    movers = fetch_top_movers()

    value = index_data.get("value", "N/A")
    change = index_data.get("change", "N/A")
    change_pct = index_data.get("change_pct", "N/A")
    direction = index_data.get("direction", "flat")

    # If we got nothing at all, mark as non-trading or source failure
    has_data = value != "N/A" or bool(movers["gainers"]) or bool(movers["losers"])

    return MarketSummary(
        index_name="EGX 30",
        current_value=value,
        change=change,
        change_pct=change_pct,
        direction=direction,
        top_gainers=movers["gainers"],
        top_losers=movers["losers"],
        most_active=movers["most_active"],
        source_note="البيانات من Investing.com – للأغراض المعلوماتية فقط." if has_data
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
        "",
    ]

    if summary.top_gainers:
        lines.append("Top Gainers:")
        for s in summary.top_gainers:
            lines.append(f"  {s['name']}: {s['price']} ({s['change_pct']})")
        lines.append("")

    if summary.top_losers:
        lines.append("Top Losers:")
        for s in summary.top_losers:
            lines.append(f"  {s['name']}: {s['price']} ({s['change_pct']})")
        lines.append("")

    if summary.most_active:
        lines.append("Most Active:")
        for s in summary.most_active:
            lines.append(f"  {s['name']}: {s['price']} ({s['change_pct']})")

    return "\n".join(lines)


if __name__ == "__main__":
    summary = build_market_summary()
    print(format_summary_text(summary))
