"""
bot.py
------
Telegram bot — Professional EGX Stock Market Intelligence Platform.

Commands:
  /today        — daily report with AI explanation + scored stocks
  /market       — EGX 30 market overview
  /watchlist    — top buy/watch opportunities
  /stock SYMBOL — detailed analysis for a specific stock
  /help         — help message
  /start        — welcome (alias for /help)

Environment variables:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — for scheduled reports
  GEMINI_API_KEY      — from Google AI Studio (free)

Run locally:
  python bot.py

Scheduled:
  python bot.py --scheduled
  python bot.py --scheduled --force  (bypass duplicate prevention)
"""

import os
import re
import sys
import time
import logging
from datetime import datetime
import pytz

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

from fetch_egx import build_market_summary
from stock_scanner import (
    scrape_egx_stock_list, download_stock_history, save_last_report,
)
from ai_report import _escape_markdown
from breakout_strategy import (
    evaluate_breakout, format_stock_breakout_detail,
    evaluate_pre_breakout, scan_pre_breakout,
    format_pre_breakout_summary, format_pre_breakout_detail,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

REQUIRED_ENV = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"]
CAIRO_TZ = pytz.timezone("Africa/Cairo")
# NOTE: Must resolve to the egx-bot/ subfolder specifically (where this file
# lives), NOT the repo root. GITHUB_WORKSPACE in CI points to the repo ROOT
# (this repo has egx-bot/ as a subfolder alongside .github/, base44/, etc.),
# so using it here caused the flag file to be written to the wrong path —
# the workflow's cache step looks for "egx-bot/.egx_sent_today" specifically,
# so a mismatch meant the duplicate-send guard silently never worked in CI.
SENT_FLAG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".egx_sent_today")


def check_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


def _is_egx_trading_day() -> bool:
    """EGX trades Sunday–Thursday. Skip Friday(4) and Saturday(5)."""
    return datetime.now(CAIRO_TZ).weekday() not in (4, 5)


def _sanitize_error(e: Exception, token: str = "") -> str:
    """Remove bot token from error messages to prevent secret exposure in logs."""
    msg = str(e)
    if token:
        msg = msg.replace(token, "[REDACTED]")
    # Also redact any pattern that looks like a bot token (digits:alphanumeric)
    msg = re.sub(r'\d{8,12}:[A-Za-z0-9_-]{30,}', '[REDACTED]', msg)
    return msg



# ─── Duplicate Prevention ────────────────────────────────────────────────────

def _already_sent_today() -> bool:
    """Check if today's report was already sent — prevents duplicate messages."""
    try:
        with open(SENT_FLAG_FILE, "r") as f:
            last_date = f.read().strip()
            today = datetime.now(CAIRO_TZ).strftime("%Y-%m-%d")
            if last_date == today:
                logger.info(f"Report already sent today ({today}). Skipping.")
                return True
    except (FileNotFoundError, IOError):
        pass
    return False


def _mark_sent_today() -> None:
    """Mark today's report as sent."""
    today = datetime.now(CAIRO_TZ).strftime("%Y-%m-%d")
    try:
        with open(SENT_FLAG_FILE, "w") as f:
            f.write(today)
        logger.info(f"Marked report as sent for {today}.")
    except IOError as e:
        logger.warning(f"Could not write sent flag: {e}")


# ─── Rate Limiting ───────────────────────────────────────────────────────────

_last_command_time: dict[int, float] = {}
_COMMAND_COOLDOWN = 30  # seconds — prevent spam


def _check_cooldown(chat_id: int) -> bool:
    """Returns True if command is allowed, False if on cooldown."""
    now = time.time()
    if chat_id in _last_command_time and now - _last_command_time[chat_id] < _COMMAND_COOLDOWN:
        return False
    _last_command_time[chat_id] = now
    return True


# ─── Command Handlers ────────────────────────────────────────────────────────

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show available commands."""
    await update.message.reply_text(
        "🇪🇬 *بوت تحليل البورصة المصرية*\n"
        "📊 Pre-Breakout Scanner | EMA50\n\n"
        "📋 *الأوامر:*\n"
        "• /today — أسهم قريبة من الاختراق\n"
        "• /market — نظرة على مؤشر EGX 30\n"
        "• /watchlist — قائمة الأسهم القريبة\n"
        "• /stock SYMBOL — تحليل تفصيلي لسهم\n"
        "   مثال: /stock COMI\n"
        "• /help — هذه الرسالة\n\n"
        "🔍 224+ سهم | score 0-100\n"
        "🔴 قرب اختراق | 🟡 تحت التجميع\n"
        "⚠️ ليست نصيحة استثمارية",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message — same as /help."""
    await cmd_help(update, context)


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Daily report: pre-breakout scanner — finds stocks ABOUT to break out."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار 30 ثانية بين الأوامر.")
        return
    msg = await update.message.reply_text(
        "⏳ جاري مسح الأسهم القريبة من الاختراق…\n"
        "📊 Pre-Breakout Scanner | EMA50"
    )

    try:
        market_summary = build_market_summary()
        stock_list = scrape_egx_stock_list()
        
        if not stock_list:
            await msg.edit_text("❌ تعذر جلب قائمة الأسهم. تحقق من المصدر.")
            return
        
        # Pre-breakout scan (stocks approaching EMA50 breakout)
        pre_signals = scan_pre_breakout(stock_list, download_stock_history)
        
        if not pre_signals:
            await msg.edit_text("⚪ لا توجد أسهم قريبة من الاختراق اليوم.")
            return
        
        market_str = ""
        if market_summary and hasattr(market_summary, 'current_value'):
            arrow = "📈" if market_summary.direction == "up" else ("📉" if market_summary.direction == "down" else "➡️")
            _val = str(market_summary.current_value)
            _pct = str(market_summary.change_pct)
            market_str = f"🇪🇬 EGX 30: {_val} {arrow} ({_pct}%)\n\n"
        
        stocks_msg = format_pre_breakout_summary(pre_signals)
        full_msg = market_str + stocks_msg
        
        await msg.delete()
        await update.message.reply_text(full_msg[:4000], parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /today: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ. يرجى المحاولة لاحقاً.")
        except Exception:
            pass

async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """EGX 30 market overview — fast, no stock scan."""
    msg = await update.message.reply_text("⏳ جاري جلب بيانات السوق…")

    try:
        market = build_market_summary()

        if not market or market.current_value == "N/A":
            await msg.edit_text(
                "⚠️ لا تتوفر بيانات مؤشر EGX 30 حاليًا.\n"
                "قد تكون البورصة مغلقة."
            )
            return

        arrow = "📈" if market.direction == "up" else ("📉" if market.direction == "down" else "➡️")
        lines = [
            "🇪🇬 *نظرة عامة على السوق*",
            "",
            f"*مؤشر EGX 30:* {_escape_markdown(str(market.current_value))} {arrow}",
            f"*التغيير اليومي:* {_escape_markdown(str(market.change))} ({_escape_markdown(str(market.change_pct))})",
        ]
        if market.month_change_pct:
            lines.append(f"*الأداء الشهري:* {_escape_markdown(str(market.month_change_pct))}")
        if market.year_change_pct:
            lines.append(f"*الأداء السنوي:* {_escape_markdown(str(market.year_change_pct))}")
        lines += [
            "",
            "📍 المصدر: Trading Economics",
        ]

        await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error in /market: {_sanitize_error(e)}")
        try:
            await msg.edit_text("❌ تعذّر جلب بيانات السوق.")
        except Exception:
            pass


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show stocks approaching breakout (pre-breakout scanner)."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار بين الأوامر.")
        return
    msg = await update.message.reply_text("⏳ جاري البحث عن أسهم قريبة من الاختراق…")

    try:
        stock_list = scrape_egx_stock_list()
        if not stock_list:
            await msg.edit_text("❌ تعذر جلب قائمة الأسهم.")
            return
        
        signals = scan_pre_breakout(stock_list, download_stock_history)
        approaching = [s for s in signals if s.is_approaching]
        accumulating = [s for s in signals if s.is_accumulating]
        
        if not approaching and not accumulating:
            await msg.edit_text("⚪ لا توجد أسهم قريبة من الاختراق حالياً.")
            return
        
        lines = []
        if approaching:
            lines.append("🔴 *قرب الاختراق:*")
            for s in approaching:
                lines.append(f"  • {s.ticker} — {s.close:.2f} | dist: {s.distance_pct:.1f}% | score: {s.score}/100")
            lines.append("")
        
        if accumulating:
            lines.append("🟡 *تحت التجميع:*")
            for s in accumulating[:15]:
                emoji = "🔴" if s.score >= 60 else "🟡"
                lines.append(f"  {emoji} {s.ticker} — {s.close:.2f} | dist: {s.distance_pct:.1f}% | score: {s.score}/100")
            if len(accumulating) > 15:
                lines.append(f"  ... و {len(accumulating) - 15} سهم آخر")
            lines.append("")
        
        lines.append("⚙️ Pre-Breakout Scanner | EMA50")
        
        await msg.delete()
        await update.message.reply_text("\n".join(lines)[:4000], parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /watchlist: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ. يرجى المحاولة لاحقاً.")
        except Exception:
            pass

async def cmd_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed analysis for a specific stock: /stock COMI
    Shows pre-breakout if below EMA50, post-breakout if above."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار بين الأوامر.")
        return
    if not context.args:
        await update.message.reply_text(
            "📋 استخدم: /stock SYMBOL\n"
            "مثال: /stock COMI"
        )
        return

    ticker = context.args[0].strip().upper().replace(".CA", "")
    msg = await update.message.reply_text(f"⏳ جاري تحليل {ticker}…")

    try:
        df = download_stock_history(ticker, n_bars=250, retries=2)
        
        if df is None or df.empty:
            await msg.edit_text(
                f"❌ لم يتم العثور على بيانات لـ {ticker}.\n"
                "تأكد من الرمز أو أن البيانات متوفرة."
            )
            return
        
        # Try pre-breakout first (stock below EMA50)
        pre = evaluate_pre_breakout(df, ticker)
        if pre.signal in ("APPROACHING", "ACCUMULATING", "NO_SETUP") and not pre.error:
            detail = format_pre_breakout_detail(pre)
        else:
            # Stock above EMA — show post-breakout analysis
            signal = evaluate_breakout(df, ticker)
            detail = format_stock_breakout_detail(signal)
        
        await msg.delete()
        await update.message.reply_text(detail, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /stock: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ أثناء التحليل.")
        except Exception:
            pass


# ─── Scheduled Push ──────────────────────────────────────────────────────────



# ─── Recommendation Tracking ────────────────────────────────────────────────

def save_recommendations_json(stocks: list, report_date: str) -> str:
    """
    Save today's recommendations and current prices to a JSON file.
    This file is committed to the repo and processed by a Base44 automation
    to track recommendation performance over time (7-day and 30-day evaluation).
    
    Returns the path to the saved JSON file.
    """
    import json
    from pathlib import Path
    
    recommendations = []
    current_prices = {}
    
    for s in stocks:
        # Current price for all stocks (used for evaluating past recommendations)
        if s.current_price > 0:
            current_prices[s.ticker] = round(s.current_price, 2)
        
        # Save actionable recommendations (Buy, Watch, Sell) + excluded stocks for analytics
        if s.scoring_result and s.scoring_result.recommendation in ("Buy", "Watch", "Sell"):
            sr = s.scoring_result
            recommendations.append({
                "ticker": s.ticker,
                "name_ar": s.name_ar,
                "score": sr.total_score,
                "price": round(s.current_price, 2),
                "type": sr.recommendation,
                "risk_level": sr.risk_level,
                # v2 risk management fields
                "stop_loss": round(sr.stop_loss, 2) if sr.stop_loss else 0.0,
                "target": round(sr.target, 2) if sr.target else 0.0,
                "rr_ratio": round(sr.rr_ratio, 2) if sr.rr_ratio else 0.0,
            })
        elif s.scoring_result and s.scoring_result.exclusion_reason:
            # Track excluded stocks for filter analytics
            sr = s.scoring_result
            recommendations.append({
                "ticker": s.ticker,
                "name_ar": s.name_ar,
                "score": 0,
                "price": round(s.current_price, 2),
                "type": "Excluded",
                "exclusion_reason": sr.exclusion_reason,
                "stop_loss": 0.0,
                "target": 0.0,
                "rr_ratio": 0.0,
            })
    
    # Count by recommendation type (exclude "Excluded" from actionable counts)
    buy_count      = sum(1 for r in recommendations if r["type"] == "Buy")
    watch_count    = sum(1 for r in recommendations if r["type"] == "Watch")
    sell_count     = sum(1 for r in recommendations if r["type"] == "Sell")
    excluded_count = sum(1 for r in recommendations if r["type"] == "Excluded")
    no_trade_count = len(stocks) - buy_count - watch_count - sell_count - excluded_count

    data = {
        "report_date": report_date,
        "recommendations": recommendations,
        "current_prices": current_prices,
        "total_stocks_scanned": len(stocks),
        "total_recommendations": buy_count + watch_count + sell_count,
        "buy_count": buy_count,
        "watch_count": watch_count,
        "sell_count": sell_count,
        "excluded_count": excluded_count,
        "no_trade_count": no_trade_count,
    }
    
    # Save to data/ directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"recommendations_{report_date}.json"
    
    def _json_default(obj):
        """Convert numpy types to native Python for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
    
    logger.info(f"Saved {len(recommendations)} recommendations to {output_path}")
    return str(output_path)


def save_pre_breakout_recommendations(signals, report_date: str) -> str:
    """Save pre-breakout scanner recommendations to JSON file."""
    import json
    from pathlib import Path

    recommendations = []
    current_prices = {}

    for s in signals:
        if hasattr(s, 'close') and s.close > 0:
            current_prices[s.ticker] = round(s.close, 2)

        if hasattr(s, 'score') and s.signal in ("APPROACHING", "ACCUMULATING"):
            recommendations.append({
                "ticker": s.ticker,
                "score": s.score,
                "price": round(s.close, 2),
                "type": s.signal,
                "distance_pct": round(s.distance_pct, 2),
                "ema_value": round(s.ema_value, 2),
                "consolidation_pct": round(s.consolidation_pct, 2),
                "volume_trend": s.volume_trend,
                "rsi": round(s.rsi, 1),
                "adx": round(s.adx, 1),
                "higher_lows": s.higher_lows,
                "stop_loss": round(s.stop_loss, 2) if s.stop_loss else 0,
                "target": round(s.target, 2) if s.target else 0,
            })
        elif hasattr(s, 'conditions_met') and s.signal in ("BUY", "WAIT"):
            # Post-breakout signals (fallback)
            recommendations.append({
                "ticker": s.ticker,
                "score": s.conditions_met * 12,
                "price": round(s.close, 2),
                "type": s.signal,
                "breakout_high": round(s.breakout_high, 2) if s.breakout_high else 0,
                "stop_loss": round(s.stop_loss, 2) if s.stop_loss else 0,
                "target": round(s.target, 2) if s.target else 0,
                "rsi": round(s.rsi, 1),
                "adx": round(s.adx, 1),
                "conditions_met": s.conditions_met,
            })

    approaching_count = sum(1 for r in recommendations if r["type"] == "APPROACHING")
    accumulating_count = sum(1 for r in recommendations if r["type"] == "ACCUMULATING")
    buy_count = sum(1 for r in recommendations if r["type"] == "BUY")
    wait_count = sum(1 for r in recommendations if r["type"] == "WAIT")

    data = {
        "report_date": report_date,
        "strategy": "Pre-Breakout Scanner",
        "recommendations": recommendations,
        "current_prices": current_prices,
        "total_recommendations": len(recommendations),
        "approaching_count": approaching_count,
        "accumulating_count": accumulating_count,
        "buy_count": buy_count,
        "wait_count": wait_count,
    }

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"recommendations_{report_date}.json"

    def _json_default(obj):
        """Convert numpy types to native Python for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)

    logger.info(f"Saved {len(recommendations)} recommendations to {output_path}")
    return str(output_path)


async def send_scheduled_report(force: bool = False) -> bool:
    """
    Push daily report to TELEGRAM_CHAT_ID (called by GitHub Actions).
    Includes duplicate prevention and automatic retry logic.
    Returns True if sent, False if skipped.
    Exits with code 1 on failure (triggers workflow retry).
    """
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID is not set.")
        sys.exit(1)

    # Skip non-trading days (Friday/Saturday in Cairo)
    if not _is_egx_trading_day():
        logger.info("Not an EGX trading day (Friday/Saturday). Skipping.")
        return False

    # Duplicate prevention — skip if already sent today
    if not force and _already_sent_today():
        return False

    bot = Bot(token=token)
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"─── Sending scheduled report (attempt {attempt}/{max_retries}) ───")

            # 1. Fetch market data
            logger.info("Step 1/5: Fetching EGX market data...")
            market_summary = build_market_summary()

            # 2. Scan all stocks using Pre-Breakout Scanner
            logger.info("Step 2/5: Scanning EGX stocks (Pre-Breakout Scanner)...")
            stock_list = scrape_egx_stock_list()
            if not stock_list:
                try:
                    await bot.send_message(chat_id=chat_id, text="❌ تعذر جلب قائمة الأسهم اليوم.")
                except Exception:
                    pass
                return False

            logger.info(f"  Got {len(stock_list)} stocks. Running pre-breakout scan...")
            pre_signals = scan_pre_breakout(stock_list, download_stock_history, max_workers=3, per_stock_timeout=25)
            logger.info(f"  Scan complete: {len(pre_signals)} pre-breakout setups found")

            # 5-star rating system — only send 4+ star stocks
            five_star, all_evaluated = find_5star_stocks(pre_signals, download_stock_history)
            logger.info(f"  5-star evaluation: {len(five_star)} stocks with 4+ stars (out of {len(all_evaluated)} evaluated)")

            if five_star:
                # Send each 5-star stock as a separate message (Yasseen's format)
                stocks_msg = ""  # Will be sent individually
            elif all_evaluated:
                # Send best available even if < 4 stars
                best = all_evaluated[0]
                stocks_msg = _format_5star_message(best)
            else:
                stocks_msg = f"⚪ لا توجد أسهم تستحق التوصية اليوم.\nتم فحص {len(stock_list)} سهم.\nسيتم المتابعة غداً إن شاء الله."

            # 3. Build market context
            logger.info("Step 3/5: Building market context...")
            market_str = ""
            if market_summary and hasattr(market_summary, 'current_value'):
                arrow = "📈" if market_summary.direction == "up" else ("📉" if market_summary.direction == "down" else "➡️")
                _val = str(market_summary.current_value)
                _chg = str(market_summary.change)
                _pct = str(market_summary.change_pct)
                market_str = f"📊 EGX 30: {_val} {arrow} ({_chg}, {_pct})"

            # 4. Send to Telegram
            logger.info("Step 4/5: Sending to Telegram...")
            messages_sent = 0
            try:
                # Send market summary first
                if market_str and len(market_str) > 10:
                    await bot.send_message(chat_id=chat_id, text=market_str[:3800])
                    messages_sent += 1

                # Send 5-star stocks individually (Yasseen's format)
                if five_star:
                    for stock in five_star[:3]:  # Max 3 stocks
                        msg = _format_5star_message(stock)
                        await bot.send_message(chat_id=chat_id, text=msg[:3800])
                        messages_sent += 1
                        logger.info(f"  Sent {stock['ticker']} ({stock['stars']}⭐)")
                elif stocks_msg and len(stocks_msg) > 10:
                    await bot.send_message(chat_id=chat_id, text=stocks_msg[:3800], parse_mode="Markdown")
                    messages_sent += 1
            except Exception as send_err:
                logger.error(f"Telegram send error: {_sanitize_error(send_err, token)}")
                raise

            # Save report for future fallback
            save_last_report({
                "date": datetime.now(CAIRO_TZ).strftime("%Y-%m-%d"),
                "ai_summary": market_str,
                "stocks_table": stocks_msg if len(stocks_msg) > 50 else "",
                "market_value": str(market_summary.current_value) if market_summary else "N/A",
                "market_change": str(market_summary.change) if market_summary else "N/A",
            })

            # Save recommendations for performance tracking
            report_date = datetime.now(CAIRO_TZ).strftime("%Y-%m-%d")
            try:
                signals_to_save = five_star if five_star else all_evaluated[:5]
                save_pre_breakout_recommendations(
                    [s['signal'] for s in signals_to_save], report_date
                )
            except Exception as e:
                logger.warning(f"Failed to save recommendations JSON: {e}")

            # Mark as sent — prevents duplicates on retry
            _mark_sent_today()
            logger.info("✅ Scheduled report sent successfully.")
            return True

        except Exception as e:
            safe_err = _sanitize_error(e, token)
            logger.error(f"❌ Attempt {attempt}/{max_retries} failed: {safe_err}")
            if attempt < max_retries:
                wait = 30 * attempt
                logger.info(f"⏳ Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                logger.error("❌ All retry attempts exhausted.")
                # Silent failure — GitHub Actions workflow already sends Telegram notification
                logger.error("❌ All retry attempts exhausted. Exiting silently.")
                sys.exit(1)

    return False


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    check_env()

    if "--scheduled" in sys.argv:
        import asyncio
        force = "--force" in sys.argv
        asyncio.run(send_scheduled_report(force=force))
        return

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    app.add_handler(CommandHandler("stock", cmd_stock))
    # Backward-compatible aliases
    app.add_handler(CommandHandler("report", cmd_today))
    app.add_handler(CommandHandler("scan", cmd_watchlist))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)




# ─── 5-Star Rating System ────────────────────────────────────────────────────


# ─── 5-Star Rating System ────────────────────────────────────────────────────

def _compute_support_resistance(df):
    """
    Compute support and resistance levels from price data.
    
    CRITICAL FIXES vs original:
    1. Support MUST be below current price, resistance MUST be above.
    2. Noise filter: S/R levels within 0.5% of current price are ignored
       (flat periods create dozens of identical swing highs/lows that are
       meaningless as support/resistance).
    3. Deduplicate swing highs/lows (flat periods create repeats).
    
    Returns: (current_price, support, resistance)
      support: nearest valid support BELOW current price (or None if none found)
      resistance: nearest valid resistance ABOVE current price (or None if none found)
    """
    if df is None or len(df) < 20:
        return None, None, None

    current_price = float(df['Close'].iloc[-1])
    recent = df.tail(60)

    # Noise threshold: levels within 0.5% of current price are not meaningful S/R
    noise_threshold = current_price * 0.01  # 1% — filters minor price noise

    # ── Find ALL swing lows and swing highs ──
    lows = recent['Low'].values
    highs = recent['High'].values

    swing_lows = []
    for i in range(2, len(lows) - 2):
        if lows[i] == min(lows[i-2:i+3]):
            swing_lows.append(float(lows[i]))

    swing_highs = []
    for i in range(2, len(highs) - 2):
        if highs[i] == max(highs[i-2:i+3]):
            swing_highs.append(float(highs[i]))

    # Deduplicate (flat periods create identical swing levels)
    swing_lows = sorted(set(swing_lows), reverse=True)   # highest first
    swing_highs = sorted(set(swing_highs))                # lowest first

    # ── Support: nearest swing low BELOW current price (at least 0.5% below) ──
    valid_supports = [s for s in swing_lows if s < current_price - noise_threshold]
    if valid_supports:
        # Use the highest valid support (closest to current price from below)
        support = valid_supports[0]  # already sorted highest-first
    else:
        # No swing low below price — use recent low as fallback
        recent_low = float(recent['Low'].min())
        support = recent_low if recent_low < current_price - noise_threshold else None

    # ── Resistance: nearest swing high ABOVE current price (at least 0.5% above) ──
    valid_resistances = [r for r in swing_highs if r > current_price + noise_threshold]
    if valid_resistances:
        # Use the lowest valid resistance (closest to current price from above)
        resistance = valid_resistances[0]  # already sorted lowest-first
    else:
        # No swing high above price — use EMA50 as resistance if price is below it
        if len(df) >= 50:
            ema50 = df['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
            resistance = float(ema50) if ema50 > current_price + noise_threshold else None
        else:
            resistance = None
        # If still no resistance, use recent high
        if resistance is None:
            recent_high = float(recent['High'].max())
            resistance = recent_high if recent_high > current_price + noise_threshold else None


    # ── EMA50 as resistance only if NO swing high resistance was found ──
    # (EMA50 should not override a stronger swing high resistance)
    if resistance is None and len(df) >= 50:
        ema50_val = float(df["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        if current_price < ema50_val - noise_threshold:
            resistance = ema50_val

    return current_price, support, resistance


def _rate_5star(signal, df):
    """
    Rate a stock 0-5 stars based on breakout readiness.
    Only 4-5 star stocks are considered "five-star" quality.
    
    CRITICAL FIXES:
    - Reject stocks where RSI is falling AND ADX is falling (no momentum + no trend)
    - Require R/R >= 2.0 (not just 1.5) for 4+ star stocks
    - Support MUST be below price, target MUST be above price
    - If risk management is invalid, stock gets 0 stars and is filtered out
    """
    stars = 0
    reasons = []
    reject_reason = None

    # ── HARD REJECT: Both RSI and ADX falling = no momentum and no trend ──
    if signal.rsi_trend == "falling" and signal.adx_trend == "falling":
        reject_reason = "RSI و ADX كلاهما هابط — لا يوجد زخم أو اتجاه"
        return {
            'ticker': signal.ticker,
            'stars': 0,
            'reasons': [],
            'price': float(df['Close'].iloc[-1]) if df is not None else 0,
            'support': None,
            'resistance': None,
            'target': None,
            'stop_loss': None,
            'rr_ratio': None,
            'entry_price': None,
            'signal': signal,
            'reject_reason': reject_reason,
        }

    # ── HARD REJECT: ADX too low (< 15) = no trend at all ──
    if signal.adx < 15:
        reject_reason = f"ADX ضعيف جداً ({signal.adx:.0f}) — لا يوجد اتجاه"
        return {
            'ticker': signal.ticker,
            'stars': 0,
            'reasons': [],
            'price': float(df['Close'].iloc[-1]) if df is not None else 0,
            'support': None,
            'resistance': None,
            'target': None,
            'stop_loss': None,
            'rr_ratio': None,
            'entry_price': None,
            'signal': signal,
            'reject_reason': reject_reason,
        }

    # Star 1: Distance from EMA50 (must be very close)
    if abs(signal.distance_pct) <= 1.5:
        stars += 1
        reasons.append(f"السعر ملاصق لـ EMA50 ({signal.distance_pct:+.1f}%)")
    elif abs(signal.distance_pct) <= 3.0:
        stars += 1
        reasons.append(f"السعر قريب من EMA50 ({signal.distance_pct:+.1f}%)")

    # Star 2: Volume rising (accumulation happening)
    if signal.volume_trend == "rising" and signal.volume_ratio >= 1.3:
        stars += 1
        reasons.append(f"حجم تداول صاعد قوي ({signal.volume_ratio:.1f}x المتوسط)")
    elif signal.volume_trend == "rising":
        stars += 1
        reasons.append(f"حجم تداول في تصاعد ({signal.volume_ratio:.1f}x)")

    # Star 3: RSI in optimal zone (45-60, rising = momentum building)
    if 45 <= signal.rsi <= 60 and signal.rsi_trend == "rising":
        stars += 1
        reasons.append(f"RSI في المنطقة الذهبية ({signal.rsi:.0f} صاعد)")
    elif 40 <= signal.rsi <= 65 and signal.rsi_trend == "rising":
        stars += 1
        reasons.append(f"RSI صاعد ({signal.rsi:.0f})")
    elif 45 <= signal.rsi <= 60 and signal.rsi_trend == "flat":
        # RSI stable in good zone — half credit but no star
        reasons.append(f"RSI ثابت في منطقة جيدة ({signal.rsi:.0f})")

    # Star 4: ADX shows trend strength
    if signal.adx >= 20 and signal.adx_trend == "rising":
        stars += 1
        reasons.append(f"ADX قوي وصاعد ({signal.adx:.0f}) — اتجاه يتكون")
    elif signal.adx >= 25:
        stars += 1
        reasons.append(f"ADX قوي ({signal.adx:.0f})")

    # Star 5: Higher lows (structural support confirmed)
    if signal.higher_lows:
        stars += 1
        reasons.append("قيعان صاعدة — دعم هيكلي مؤكد")

    # ── Compute support/resistance and risk management ──
    current_price, support, resistance = _compute_support_resistance(df)

    target = None
    stop_loss = None
    rr_ratio = None
    entry_price = None

    # CRITICAL: Validate support is below price and resistance is above
    if support is not None and resistance is not None and current_price is not None:
        if support >= current_price:
            # Support above price = invalid, don't compute risk
            reject_reason = "الدعم فوق السعر — بنية سعيرية غير صالحة"
        elif resistance <= current_price:
            # Resistance below price = invalid
            reject_reason = "المقاومة تحت السعر — بنية سعيرية غير صالحة"
        else:
            # Valid support/resistance — calculate risk management
            entry_price = current_price
            stop_loss = support * 0.99  # 1% below support
            risk = entry_price - stop_loss
            # Target = resistance, with minimum R/R of 2:1
            reward_to_resistance = resistance - entry_price
            rr_to_resistance = reward_to_resistance / risk if risk > 0 else 0

            if rr_to_resistance >= 2.0:
                # Resistance is far enough — use it as target
                target = resistance
                rr_ratio = rr_to_resistance
            elif rr_to_resistance >= 1.5:
                # Resistance is okay but not great — extend target beyond resistance
                target = entry_price + (risk * 2.0)  # Force 2:1 R/R
                rr_ratio = 2.0
            else:
                # Resistance too close — skip this stock
                reject_reason = f"المقاومة قريبة جداً (R/R = {rr_to_resistance:.1f})"

            if target and stop_loss and target <= stop_loss:
                reject_reason = "الهدف تحت الستوب لوس — حساب خاطئ"
                target = None
                stop_loss = None
                rr_ratio = None

    if reject_reason:
        return {
            'ticker': signal.ticker,
            'stars': 0,
            'reasons': [],
            'price': current_price or 0,
            'support': support,
            'resistance': resistance,
            'target': None,
            'stop_loss': None,
            'rr_ratio': None,
            'entry_price': None,
            'signal': signal,
            'reject_reason': reject_reason,
        }

    return {
        'ticker': signal.ticker,
        'stars': stars,
        'reasons': reasons,
        'price': current_price,
        'support': support,
        'resistance': resistance,
        'target': target,
        'stop_loss': stop_loss,
        'rr_ratio': rr_ratio,
        'entry_price': entry_price,
        'signal': signal,
        'reject_reason': None,
    }


def _format_5star_message(stock):
    """Format a 5-star stock recommendation as Yasseen requested."""
    s = stock['signal']
    stars_str = "⭐" * max(stock['stars'], 1)

    p = f"{stock['price']:.2f}" if stock['price'] else "N/A"
    sup = f"{stock['support']:.2f}" if stock['support'] else "N/A"
    res = f"{stock['resistance']:.2f}" if stock['resistance'] else "N/A"
    tgt = f"{stock['target']:.2f}" if stock['target'] else "N/A"
    sl = f"{stock['stop_loss']:.2f}" if stock['stop_loss'] else "N/A"
    rr = f"{stock['rr_ratio']:.1f}:1" if stock['rr_ratio'] else "N/A"
    entry = f"{stock['entry_price']:.2f}" if stock['entry_price'] else p

    # Validate: target must be above entry, stop must be below entry
    if stock['target'] and stock['stop_loss'] and stock['entry_price']:
        if stock['target'] <= stock['entry_price']:
            tgt = "⚠️ غير صالح"
        if stock['stop_loss'] >= stock['entry_price']:
            sl = "⚠️ غير صالح"

    msg = f"{stars_str}\n"
    msg += f"📊 {stock['ticker']}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"💰 سعر الدخول: {entry} EGP\n"
    msg += f"🟢 الدعم: {sup} EGP\n"
    msg += f"🔴 المقاومة: {res} EGP\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += f"🎯 الهدف: {tgt} EGP\n"
    msg += f"🛑 ستوب لوس: {sl} EGP\n"
    msg += f"📈 المخاطرة/العائد: {rr}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "📋 ليه هذا السهم؟\n"
    for r in stock['reasons']:
        msg += f"  ✅ {r}\n"
    if not stock['reasons']:
        msg += f"  • Score: {s.score}/100\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "📊 المؤشرات:\n"
    msg += f"  • المسافة من EMA50: {s.distance_pct:+.1f}%\n"
    msg += f"  • RSI: {s.rsi:.0f} ({s.rsi_trend})\n"
    msg += f"  • ADX: {s.adx:.0f} ({s.adx_trend})\n"
    msg += f"  • الحجم: {s.volume_trend} ({s.volume_ratio:.1f}x)\n"
    msg += f"  • قيعان صاعدة: {'نعم ✅' if s.higher_lows else 'لا ❌'}\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "📌 شروط الدخول:\n"
    msg += "  • اختراق EMA50 بإغلاق يومي فوقها\n"
    msg += "  • حجم تداول أعلى من المتوسط\n"
    msg += "  • RSI فوق 50\n"
    msg += "━━━━━━━━━━━━━━━\n"
    msg += "⚠️ هذه ليست نصيحة مالية. التداول ينطوي على مخاطر."

    return msg


def find_5star_stocks(signals, download_func):
    """
    Evaluate pre-breakout signals and return only 4+ star stocks.
    Uses scan cache to avoid re-downloading (prevents 429 rate limits).
    
    CRITICAL FIXES:
    - Filter out stocks with None/negative R/R ratio (not just < 1.5)
    - Filter out stocks with reject_reason (invalid support/resistance/etc.)
    - Filter out penny stocks (price < 1 EGP)
    - Require R/R >= 2.0 for 4+ star stocks
    - Require valid entry_price (not None)
    """
    from stock_scanner import _df_cache
    evaluated = []
    rejected = []

    for signal in signals:
        if signal.score < 40:
            continue
        # Try cache first to avoid extra TradingView requests (prevents 429s)
        df = _df_cache.get(signal.ticker)
        if df is None:
            df = download_func(signal.ticker, n_bars=150, retries=1)
        if df is None or len(df) < 50:
            continue

        result = _rate_5star(signal, df)

        # Skip rejected stocks (invalid support/resistance, weak indicators, etc.)
        if result.get('reject_reason'):
            rejected.append((signal.ticker, result['reject_reason']))
            continue

        # Filter out penny stocks (price < 1 EGP)
        if result['price'] and result['price'] < 1.0:
            continue

        # CRITICAL: Filter out stocks with no valid support/resistance
        if result['support'] is None or result['resistance'] is None:
            continue

        # CRITICAL: Filter out stocks with no valid R/R ratio (was passing before!)
        if result['rr_ratio'] is None or result['rr_ratio'] < 1.5:
            continue

        # CRITICAL: Filter out stocks with no entry price
        if result['entry_price'] is None:
            continue

        # CRITICAL: Validate target > entry > stop_loss
        if result['target'] and result['stop_loss'] and result['entry_price']:
            if result['target'] <= result['entry_price']:
                continue
            if result['stop_loss'] >= result['entry_price']:
                continue

        evaluated.append(result)

    # Log rejected stocks for transparency
    if rejected:
        for ticker, reason in rejected[:10]:
            logger.info(f"  ❌ {ticker}: {reason}")

    # Sort by stars (desc), then by R/R ratio (desc), then by score (desc)
    evaluated.sort(key=lambda x: (x['stars'], x.get('rr_ratio', 0) or 0, x['signal'].score), reverse=True)

    # Return only 4+ star stocks
    return [s for s in evaluated if s['stars'] >= 4], evaluated

if __name__ == "__main__":
    main()


