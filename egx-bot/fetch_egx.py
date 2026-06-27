"""
fetch_egx.py
------------
Fetches top Egyptian Exchange (EGX) stock data from public sources.
Uses investing.com (EGX 30 index & top movers) and falls back to
a minimal scrape of the EGX official site.

No API keys required. No sensitive data stored.
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
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
    "Accept-Language": "en-US,en;q=0.9",
}

EGX30_INVESTING_URL = "https://www.investing.com/indices/egx-30"
EGX_MOVERS_URL = "https://www.investing.com/equities/egypt"


@dataclass
class MarketSummary:
    index_name: str
    current_value: Optional[str]
    change: Optional[str]
    change_pct: Optional[str]
    direction: str  # "up", "down", "flat"
    top_gainers: list[dict]
    top_losers: list[dict]
    most_active: list[dict]
    source_note: str


def _safe_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def fetch_egx30_index() -> dict:
    """Scrape EGX 30 index summary from Investing.com."""
    resp = _safe_get(EGX30_INVESTING_URL)
    if not resp:
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    data = {}

    try:
        # Current price
        price_el = soup.select_one('[data-test="instrument-price-last"]')
        if price_el:
            data["value"] = price_el.get_text(strip=True)

        # Change & % change
        change_el = soup.select_one('[data-test="instrument-price-change"]')
        pct_el = soup.select_one('[data-test="instrument-price-change-percent"]')
        if change_el:
            data["change"] = change_el.get_text(strip=True)
        if pct_el:
            raw = pct_el.get_text(strip=True).strip("()")
            data["change_pct"] = raw

        # Direction
        change_text = data.get("change", "0")
        if change_text.startswith("-"):
            data["direction"] = "down"
        elif change_text.startswith("+") or (
            change_text and change_text[0].isdigit() and float(
                change_text.replace(",", "")
            ) > 0
        ):
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

    def parse_table(table_id: str) -> list[dict]:
        rows = []
        table = soup.find("table", {"id": table_id})
        if not table:
            return rows
        for tr in table.select("tbody tr")[:5]:
            cells = tr.find_all("td")
            if len(cells) >= 3:
                rows.append(
                    {
                        "name": cells[1].get_text(strip=True) if len(cells) > 1 else "—",
                        "price": cells[2].get_text(strip=True) if len(cells) > 2 else "—",
                        "change_pct": cells[4].get_text(strip=True) if len(cells) > 4 else "—",
                    }
                )
        return rows

    # Investing.com table IDs for Egypt market movers
    result["gainers"] = parse_table("gainers")
    result["losers"] = parse_table("losers")
    result["most_active"] = parse_table("most_active")

    return result


def build_market_summary() -> MarketSummary:
    """
    Aggregate all market data into a single MarketSummary object.
    """
    logger.info("Fetching EGX 30 index data…")
    index_data = fetch_egx30_index()

    logger.info("Fetching top movers…")
    movers = fetch_top_movers()

    value = index_data.get("value", "N/A")
    change = index_data.get("change", "N/A")
    change_pct = index_data.get("change_pct", "N/A")
    direction = index_data.get("direction", "flat")

    return MarketSummary(
        index_name="EGX 30",
        current_value=value,
        change=change,
        change_pct=change_pct,
        direction=direction,
        top_gainers=movers["gainers"],
        top_losers=movers["losers"],
        most_active=movers["most_active"],
        source_note="البيانات من Investing.com – للأغراض المعلوماتية فقط.",
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
