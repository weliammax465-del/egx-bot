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

from fetch_egx import build_market_summary, format_summary_text
from stock_scanner import (
    scan_all_stocks, format_analysis_for_ai,
    get_buy_signals, get_watchlist,
    scan_single_stock, scrape_egx_stock_list, download_stock_history,
    get_scan_status, load_last_report, save_last_report,
)
from ai_report import (
    explain_analysis, build_telegram_message, build_stocks_table_message,
    format_stock_detail, _escape_markdown, _safe_truncate,
)
from breakout_strategy import (
    evaluate_breakout, scan_stocks_breakout,
    format_breakout_summary, format_stock_breakout_detail,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

REQUIRED_ENV = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"]
CAIRO_TZ = pytz.timezone("Africa/Cairo")
SENT_FLAG_FILE = os.path.join(os.environ.get("GITHUB_WORKSPACE", os.path.dirname(__file__)), ".egx_sent_today")


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
        "📊 استراتيجية اختراق EMA50\n\n"
        "📋 *الأوامر:*\n"
        "• /today — تقرير يومي (إشارات الشراء والانتظار)\n"
        "• /market — نظرة على مؤشر EGX 30\n"
        "• /watchlist — أسهم في انتظار تأكيد الاختراق\n"
        "• /stock SYMBOL — تحليل تفصيلي لسهم\n"
        "   مثال: /stock COMI\n"
        "• /help — هذه الرسالة\n\n"
        "🔍 224+ سهم | 8 شروط للتأكيد\n"
        "⚙️ EMA50 + RSI + ADX + Volume\n"
        "⚠️ ليست نصيحة استثمارية",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message — same as /help."""
    await cmd_help(update, context)


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Daily report: market overview + EMA50 breakout strategy scan."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار 30 ثانية بين الأوامر.")
        return
    msg = await update.message.reply_text(
        "⏳ جاري تحليل أسهم البورصة المصرية…\n"
        "📊 استراتيجية اختراق EMA50 | 8 شروط للتأكيد"
    )

    try:
        market_summary = build_market_summary()
        stock_list = scrape_egx_stock_list()
        
        if not stock_list:
            await msg.edit_text("❌ تعذر جلب قائمة الأسهم. تحقق من المصدر.")
            return
        
        signals = scan_stocks_breakout(stock_list, download_stock_history)
        
        if not signals:
            await msg.edit_text("❌ لا توجد بيانات كافية اليوم.")
            return
        
        market_str = ""
        if market_summary and hasattr(market_summary, 'current_value'):
            arrow = "📈" if market_summary.direction == "up" else ("📉" if market_summary.direction == "down" else "➡️")
            market_str = f"🇪🇬 EGX 30: {market_summary.current_value:,} {arrow} ({market_summary.change_pct}%)" + "\n\n"
        
        stocks_msg = format_breakout_summary(signals)
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
            f"📍 المصدر: Trading Economics",
        ]

        await msg.edit_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error in /market: {_sanitize_error(e)}")
        try:
            await msg.edit_text("❌ تعذّر جلب بيانات السوق.")
        except Exception:
            pass


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show stocks in WAIT state (near breakout confirmation)."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار بين الأوامر.")
        return
    msg = await update.message.reply_text("⏳ جاري البحث عن أسهم في انتظار التأكيد…")

    try:
        stock_list = scrape_egx_stock_list()
        if not stock_list:
            await msg.edit_text("❌ تعذر جلب قائمة الأسهم.")
            return
        
        signals = scan_stocks_breakout(stock_list, download_stock_history)
        wait_signals = [s for s in signals if s.signal == "WAIT"]
        buy_signals = [s for s in signals if s.is_buy]
        
        if not wait_signals and not buy_signals:
            await msg.edit_text("⚪ لا توجد أسهم في انتظار التأكيد حالياً.")
            return
        
        lines = []
        if buy_signals:
            lines.append("🟢 *إشارات شراء:*")
            for s in buy_signals:
                lines.append(f"  • {s.ticker} — {s.close:.2f} | {s.conditions_met}/8 شروط")
            lines.append("")
        
        if wait_signals:
            lines.append("⏸ *في انتظار التأكيد:*")
            for s in sorted(wait_signals, key=lambda x: x.conditions_met, reverse=True):
                lines.append(f"  • {s.ticker} — {s.close:.2f} | {s.conditions_met}/8 شروط | اختراق: {s.breakout_high:.2f}")
            lines.append("")
        
        lines.append("⚙️ EMA50 Breakout Strategy")
        
        await msg.delete()
        await update.message.reply_text("\n".join(lines)[:4000], parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in /watchlist: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ. يرجى المحاولة لاحقاً.")
        except Exception:
            pass

async def cmd_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed breakout analysis for a specific stock: /stock COMI"""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار بين الأوامر.")
        return
    if not context.args:
        await update.message.reply_text(
            "📋 استخدم: /stock SYMBOL\n"
            "مثال: /stock COMI\n\n"
            "أو جرب: /stock ETEL أو /stock ORAS"
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
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(recommendations)} recommendations to {output_path}")
    return str(output_path)


def save_breakout_recommendations(signals, report_date: str) -> str:
    """
    Save breakout strategy recommendations to JSON file.
    Only saves BUY signals and WAIT (near-signal) stocks.
    """
    import json
    from pathlib import Path

    recommendations = []
    current_prices = {}

    for s in signals:
        if s.close > 0:
            current_prices[s.ticker] = round(s.close, 2)

        if s.signal in ("BUY", "WAIT"):
            recommendations.append({
                "ticker": s.ticker,
                "score": s.conditions_met * 12,  # 0-96 scale (8 conds * 12)
                "price": round(s.close, 2),
                "type": s.signal,  # "BUY" or "WAIT"
                "breakout_high": round(s.breakout_high, 2) if s.breakout_high else 0,
                "stop_loss": round(s.stop_loss, 2) if s.stop_loss else 0,
                "target": round(s.target, 2) if s.target else 0,
                "rsi": round(s.rsi, 1),
                "adx": round(s.adx, 1),
                "conditions_met": s.conditions_met,
            })

    buy_count = sum(1 for r in recommendations if r["type"] == "BUY")
    wait_count = sum(1 for r in recommendations if r["type"] == "WAIT")

    data = {
        "report_date": report_date,
        "strategy": "EMA50_Breakout",
        "recommendations": recommendations,
        "current_prices": current_prices,
        "total_stocks_scanned": len(signals),
        "total_recommendations": len(recommendations),
        "buy_count": buy_count,
        "wait_count": wait_count,
    }

    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"recommendations_{report_date}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(recommendations)} breakout recommendations ({buy_count} BUY, {wait_count} WAIT) to {output_path}")
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

            await bot.send_message(
                chat_id=chat_id,
                text="⏳ جاري تحضير التقرير اليومي…",
            )

            # 1. Fetch market data
            logger.info("Step 1/5: Fetching EGX market data...")
            market_summary = build_market_summary()

            # 2. Scan all stocks using EMA50 Breakout Strategy
            logger.info("Step 2/5: Scanning EGX stocks (EMA50 Breakout Strategy)...")
            stock_list = scrape_egx_stock_list()
            if not stock_list:
                await bot.send_message(chat_id=chat_id, text="❌ تعذر جلب قائمة الأسهم اليوم.")
                return False

            logger.info(f"  Got {len(stock_list)} stocks. Running breakout scan...")
            signals = scan_stocks_breakout(stock_list, download_stock_history)
            logger.info(f"  Scan complete: {len(signals)} stocks analyzed")

            if not signals:
                await bot.send_message(
                    chat_id=chat_id,
                    text="❌ لا توجد بيانات كافية اليوم. تحقق من مصدر TradingView.",
                )
                return False

            # 3. Build market context (skip AI — breakout summary is the main content)
            logger.info("Step 3/5: Building market context...")
            ai_summary = ""
            if market_summary and hasattr(market_summary, 'current_value'):
                arrow = "📈" if market_summary.direction == "up" else ("📉" if market_summary.direction == "down" else "➡️")
                ai_summary = f"📊 EGX 30: {market_summary.current_value:,} {arrow} ({market_summary.change:+,}, {market_summary.change_pct:+.2f}%)"

            # 4. Build Telegram messages
            logger.info("Step 4/5: Building report messages...")
            market_str = ""
            if market_summary and hasattr(market_summary, 'current_value'):
                market_str = f"📊 EGX 30: {market_summary.current_value:,} ({market_summary.change:+,}, {market_summary.change_pct:+.2f}%)\n"

            main_msg = ai_summary if ai_summary else market_str

            # Stocks message: breakout strategy results
            stocks_msg = format_breakout_summary(signals)

            # 5. Send to Telegram
            logger.info("Step 5/5: Sending to Telegram...")
            if main_msg and len(main_msg) > 10:
                await bot.send_message(chat_id=chat_id, text=main_msg[:3800])
            await bot.send_message(chat_id=chat_id, text=stocks_msg[:3800], parse_mode="Markdown")

            # Save report for future fallback
            save_last_report({
                "date": datetime.now(CAIRO_TZ).strftime("%Y-%m-%d"),
                "ai_summary": ai_summary if ai_summary else "",
                "stocks_table": stocks_msg if len(stocks_msg) > 50 else "",
                "market_value": str(market_summary.current_value) if market_summary else "N/A",
                "market_change": str(market_summary.change) if market_summary else "N/A",
            })

            # Save recommendations for performance tracking
            report_date = datetime.now(CAIRO_TZ).strftime("%Y-%m-%d")
            try:
                save_breakout_recommendations(signals, report_date)
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
                try:
                    await bot.send_message(chat_id=chat_id, text="❌ تعذّر إرسال التقرير اليوم بعد عدة محاولات.")
                except Exception:
                    pass
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


if __name__ == "__main__":
    main()
