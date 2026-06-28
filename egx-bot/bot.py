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
    scan_single_stock,
    get_scan_status, load_last_report, save_last_report,
)
from ai_report import (
    explain_analysis, build_telegram_message, build_stocks_table_message,
    format_stock_detail, _escape_markdown, _safe_truncate,
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
        "🇪🇬 *بوت التحليل التقني للبورصة المصرية*\n\n"
        "📊 *الأوامر المتاحة:*\n"
        "• /today — تقرير يومي كامل بالدرجات والتوصيات\n"
        "• /market — نظرة عامة على مؤشر EGX 30\n"
        "• /watchlist — أقوى فرص الشراء والمراقبة\n"
        "• /stock SYMBOL — تحليل تفصيلي لسهم معين\n"
        "   مثال: /stock COMI\n"
        "• /help — هذه الرسالة\n\n"
        "🔬 يتم تحليل 224+ سهم باستخدام 15+ مؤشر تقني\n"
        "🎯 درجة محسوبة من 0-100 لكل سهم\n"
        "📋 توصية: شراء / مراقبة / بيع / لا تداول",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message — same as /help."""
    await cmd_help(update, context)


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Full daily report: market overview + AI explanation + scored stocks."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار 30 ثانية بين الأوامر.")
        return
    msg = await update.message.reply_text(
        "⏳ جاري تحليل جميع أسهم البورصة المصرية…\n"
        "📊 حساب 15+ مؤشر تقني + درجة من 100 لكل سهم"
    )

    try:
        market_summary = build_market_summary()
        stocks = scan_all_stocks()

        if not stocks:
            await msg.edit_text(
                "⚠️ لا تتوفر بيانات موثقة اليوم.\n"
                "قد تكون البورصة مغلقة أو المصدر غير متاح."
            )
            return

        market_text = format_summary_text(market_summary) if market_summary else ""
        computed_data = format_analysis_for_ai(stocks, market_text)
        ai_summary = explain_analysis(computed_data)

        main_msg = build_telegram_message(ai_summary, stocks, market_summary)
        await msg.delete()
        await update.message.reply_text(main_msg, parse_mode=ParseMode.MARKDOWN)

        stocks_msg = build_stocks_table_message(stocks)
        if len(stocks_msg) > 50:
            await update.message.reply_text(stocks_msg, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error in /today: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ. يرجى المحاولة لاحقًا.")
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
    """Top buy and watch opportunities only — no AI summary."""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار 30 ثانية بين الأوامر.")
        return
    msg = await update.message.reply_text("⏳ جاري مسح الأسهم وتحديد التوصيات…")

    try:
        stocks = scan_all_stocks()

        if not stocks:
            await msg.edit_text(
                "⚪ لا توجد فرص تداول موثقة اليوم.\n"
                "البيانات غير كافية أو السوق مغلق."
            )
            return

        buy = get_buy_signals(stocks)
        watch = get_watchlist(stocks)

        # Filter out Buy stocks from Watch (avoid duplicates)
        buy_tickers = {s.ticker for s in buy}
        watch_only = [s for s in watch if s.ticker not in buy_tickers]

        if not buy and not watch_only:
            await msg.edit_text(
                "⚪ لا توجد فرص تداول موثقة اليوم.\n"
                "لا توجد أسهم بدرجة شراء أو مراقبة."
            )
            return

        lines = ["🎯 *قائمة المراقبة*", ""]

        if buy:
            lines.append("🟢 *شراء (درجة 70+):*")
            lines.append("")
            for i, s in enumerate(buy[:10], 1):
                sr = s.scoring_result
                lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
                lines.append(f"   {_escape_markdown(str(s.current_price))} EGP | درجة: {sr.total_score}/100")
                if sr.pass_reasons:
                    lines.append(f"   ✅ {_escape_markdown(sr.pass_reasons[0])}")
                lines.append("")

        if watch_only:
            lines.append("🟡 *مراقبة (درجة 50-69):*")
            lines.append("")
            for i, s in enumerate(watch_only[:5], 1):
                sr = s.scoring_result
                lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
                lines.append(f"   {_escape_markdown(str(s.current_price))} EGP | درجة: {sr.total_score}/100")
                lines.append("")

        lines += [
            "─────────────────────",
            "⏰ تأكد من البيانات قبل اتخاذ أي قرار",
        ]

        await msg.delete()
        await update.message.reply_text(
            _safe_truncate("\n".join(lines), 4000),
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        logger.error(f"Error in /watchlist: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ أثناء المسح.")
        except Exception:
            pass


async def cmd_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Detailed analysis for a specific stock: /stock COMI"""
    if not _check_cooldown(update.effective_chat.id):
        await update.message.reply_text("⏳ يرجى الانتظار بين الأوامر.")
        return
    if not context.args:
        await update.message.reply_text(
            "📋 استخدم: `/stock SYMBOL`\n"
            "مثال: `/stock COMI`\n\n"
            "أو جرب: `/stock ETEL` أو `/stock TMGH`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    ticker = context.args[0].strip().upper().replace(".CA", "")
    msg = await update.message.reply_text(f"⏳ جاري تحليل {ticker}…")

    try:
        stock = scan_single_stock(ticker)

        if not stock:
            await msg.edit_text(
                f"❌ لم يتم العثور على السهم {_escape_markdown(ticker)}.\n"
                "تأكد من الرمز أو أن البيانات متوفرة."
            )
            return

        detail = format_stock_detail(stock)
        await msg.delete()
        await update.message.reply_text(detail, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Error in /stock: {_sanitize_error(e)}")
        try:
            await update.message.reply_text("❌ حدث خطأ أثناء التحليل.")
        except Exception:
            pass


# ─── Scheduled Push ──────────────────────────────────────────────────────────

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
                text="⏳ جاري تحضير التقرير اليومي…\n📊 تحليل 224+ سهم | 15+ مؤشر تقني | درجة من 100",
            )

            # 1. Fetch market data
            logger.info("Step 1/5: Fetching EGX market data...")
            market_summary = build_market_summary()

            # 2. Scan all stocks (validate + download + analyze + score)
            logger.info("Step 2/5: Scanning all EGX stocks...")
            stocks = scan_all_stocks()

            if not stocks:
                # Get detailed failure info
                status = get_scan_status()
                failed_sources = ", ".join(status.failed_sources) if status.failed_sources else "غير محدد"
                
                # Try to use last successful report as fallback
                last_report = load_last_report()
                if last_report and last_report.get("ai_summary"):
                    fallback_date = last_report.get("date", "غير محدد")
                    fallback_msg = (
                        f"⚠️ تعذّر الحصول على بيانات جديدة اليوم.\n"
                        f"📋 يتم عرض آخر تقرير ناجح من يوم {fallback_date}.\n"
                        f"🔍 سبب الفشل: {failed_sources}\n\n"
                        f"{last_report.get('ai_summary', '')[:3000]}"
                    )
                    await bot.send_message(chat_id=chat_id, text=fallback_msg)
                    if last_report.get("stocks_table"):
                        await bot.send_message(chat_id=chat_id, text=last_report["stocks_table"][:3800])
                    logger.warning(f"No new data. Sent fallback report from {fallback_date}. Failed: {failed_sources}")
                else:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            f"❌ لا توجد بيانات موثوقة كافية اليوم.\n"
                            f"🔍 المصدر الفاشل: {failed_sources}\n"
                            f"⏳ سيتم إعادة المحاولة تلقائيًا."
                        ),
                    )
                    logger.warning(f"No data and no fallback. Failed sources: {failed_sources}")
                return False

            # 3. Generate AI explanation
            logger.info("Step 3/5: Generating AI analysis...")
            market_text = format_summary_text(market_summary) if market_summary else ""
            computed_data = format_analysis_for_ai(stocks, market_text)
            ai_summary = explain_analysis(computed_data)

            # 4. Build Telegram messages
            logger.info("Step 4/5: Building report messages...")
            main_msg = build_telegram_message(ai_summary, stocks, market_summary)
            stocks_msg = build_stocks_table_message(stocks)

            # 5. Send to Telegram
            logger.info("Step 5/5: Sending to Telegram...")
            await bot.send_message(chat_id=chat_id, text=main_msg, parse_mode=ParseMode.MARKDOWN)
            if len(stocks_msg) > 50:
                await bot.send_message(chat_id=chat_id, text=stocks_msg, parse_mode=ParseMode.MARKDOWN)

            # Save report for future fallback
            save_last_report({
                "date": datetime.now(CAIRO_TZ).strftime("%Y-%m-%d"),
                "ai_summary": ai_summary,
                "stocks_table": stocks_msg if len(stocks_msg) > 50 else "",
                "market_value": str(market_summary.current_value) if market_summary else "N/A",
                "market_change": str(market_summary.change) if market_summary else "N/A",
            })

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
