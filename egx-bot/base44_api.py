"""
base44_api.py
-------------
Client for the Base44 backend function `egxRecommendationApi`.

Provides two functions:
  - save_recommendations():  Save today's Buy/Watch/Sell recommendations
  - evaluate_recommendations(): Evaluate past recommendations with current prices

Called by bot.py after the daily report is sent to Telegram.
No API key required — the function uses service-role access.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ─── Configuration ──────────────────────────────────────────────────────────

_APP_ID = "6a4032e60807c2ecdbca5a98"
_FUNCTION_NAME = "egxRecommendationApi"
_API_URL = f"https://{_APP_ID}.base44.app/api/apps/{_APP_ID}/functions/{_FUNCTION_NAME}"

_TIMEOUT = 30  # seconds
_MAX_RETRIES = 2
_RETRY_DELAY = 5  # seconds


def _call_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Call the Base44 backend function with retry logic.
    Returns the JSON response dict, or None on failure.
    """
    import time

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                _API_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(
                    f"Base44 API returned {resp.status_code}: {resp.text[:200]}"
                )
        except requests.exceptions.Timeout:
            logger.warning(f"Base44 API timeout (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Base44 API connection error (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.error(f"Base44 API error (attempt {attempt + 1}): {e}")

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY)

    return None


# ─── Public API ─────────────────────────────────────────────────────────────


def save_recommendations(
    stocks: list[Any],
    report_date: str | None = None,
    report_id: str = "",
) -> bool:
    """
    Save today's recommendations to the RecommendationHistory entity.

    Args:
        stocks: List of StockAnalysis objects with scoring_result set.
        report_date: YYYY-MM-DD string. Defaults to today.
        report_id: Optional reference to the DailyReport entity.

    Returns True if saved successfully, False otherwise.
    """
    if not stocks:
        logger.info("No stocks to save recommendations for")
        return False

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    # Extract recommendations (only Buy, Watch, Sell — skip No Trade)
    valid_types = {"Buy", "Watch", "Sell"}
    recommendations = []

    for stock in stocks:
        scoring = getattr(stock, "scoring_result", None)
        if scoring is None:
            continue

        rec_type = getattr(scoring, "recommendation", "")
        if rec_type not in valid_types:
            continue

        recommendations.append({
            "ticker": stock.ticker,
            "score": getattr(scoring, "composite_score", 0),
            "price": stock.current_price,
            "type": rec_type,
        })

    if not recommendations:
        logger.info("No Buy/Watch/Sell recommendations to save")
        return True  # Not an error — just nothing to save

    payload = {
        "action": "save",
        "report_date": report_date,
        "report_id": report_id,
        "recommendations": recommendations,
    }

    result = _call_api(payload)
    if result and result.get("success"):
        saved = result.get("saved", 0)
        logger.info(f"Saved {saved} recommendations to Base44 (date={report_date})")
        return True
    elif result and result.get("message"):
        logger.info(f"Recommendations already saved: {result.get('message')}")
        return True  # Duplicate — not an error
    else:
        logger.error("Failed to save recommendations to Base44")
        return False


def evaluate_recommendations(
    stocks: list[Any],
    today: str | None = None,
) -> bool:
    """
    Evaluate past recommendations using today's stock prices.

    Called after the daily scan — sends current prices for all stocks
    to the backend function, which evaluates any recommendations that
    are due for 7-day or 30-day checks.

    Args:
        stocks: List of StockAnalysis objects with current_price set.
        today: YYYY-MM-DD string. Defaults to today.

    Returns True if evaluation completed, False otherwise.
    """
    if not stocks:
        return False

    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    # Build current prices dict
    current_prices = {}
    for stock in stocks:
        if stock.current_price and stock.current_price > 0:
            current_prices[stock.ticker] = stock.current_price

    if not current_prices:
        logger.info("No current prices to send for evaluation")
        return False

    payload = {
        "action": "evaluate",
        "today": today,
        "current_prices": current_prices,
    }

    result = _call_api(payload)
    if result and result.get("success"):
        eval_7d = result.get("evaluated_7d", 0)
        eval_30d = result.get("evaluated_30d", 0)
        total = result.get("total_evaluated", 0)
        if total > 0:
            logger.info(
                f"Evaluated {total} past recommendations "
                f"(7d: {eval_7d}, 30d: {eval_30d})"
            )
        return True
    else:
        logger.warning("Failed to evaluate past recommendations")
        return False


def get_performance_stats() -> dict[str, Any] | None:
    """
    Get performance statistics for the dashboard.
    Returns win rate, avg profit/loss, total recommendations, etc.
    """
    result = _call_api({"action": "stats"})
    if result:
        return result
    return None


def get_recommendation_history(
    ticker: str = "",
    recommendation_type: str = "",
    limit: int = 50,
    skip: int = 0,
) -> dict[str, Any] | None:
    """
    Get recommendation history records.
    """
    payload = {"action": "history", "limit": limit, "skip": skip}
    if ticker:
        payload["ticker"] = ticker
    if recommendation_type:
        payload["recommendation_type"] = recommendation_type

    result = _call_api(payload)
    if result:
        return result
    return None
