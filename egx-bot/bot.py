"""
bot.py
------
Telegram bot entry point — Professional EGX Technical Analysis.

Commands:
  /start   — welcome message
  /report  — full technical analysis report (all stocks)
  /scan    — quick scan of top bullish/bearish stocks

Environment variables required:
  TELEGRAM_BOT_TOKEN  — from @BotFather
  TELEGRAM_CHAT_ID    — your personal chat ID or channel ID
  GEMINI_API_KEY      — from Google AI Studio (free)

Run locally:
  python bot.py

For scheduled daily delivery, use GitHub Actions (see .github/workflows/daily.yml).
"""

import os
import sys
import logging
from datetime import datetime
import pytz

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode

from fetch_egx import build_market_summary, format_summary_text
from stock_scanner import scan_all_stocks, format_analysis_for_ai
from ai_report import (
    generate_arabic_report,
    build_telegram_message,
    build_stocks_table_message,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

REQUIRED_ENV = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"]
CAIRO_TZ = pytz.timezone("Africa/Cairo")


def check_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


def _is_egx_trading_day() -> bool:
    """
    EGX trades Sunday through Thursday.
    Python weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6.
    Skip Friday(4) and Saturday(5).
    """
    now = datetime.now(CAIRO_TZ)
    return now.weekday() not in (4, 5)


# ─── Command Handlers ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🇪🇬 *مرحبًا بك في بوت التحليل التقني للبورصة المصرية!*\n\n"
        "الأوامر المتاحة:\n"
        "• /report — تقرير تحليلي كامل لكل الأسهم\n"
        "• /scan — مسح سريع لأقوى الأسهم صعودًا وهبوطًا\n\n"
        "📊 يتم تحليل جميع أسهم البورصة المصرية (224+ سهم)\n"
        "🔬 باستخدام 9 مؤشرات تقنية احترافية",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Full professional technical analysis report."""
    msg = await update.message.reply_text(
        "⏳ جاري تحليل جميع أسهم البورصة المصرية…\n"
        "📊 يتم حساب: RSI، Stochastic، MACD، Bollinger، Volume Profile، والمزيد…"
    )

    try:
        market_summary = build_market_summary()
        stocks = scan_all_stocks()

        if not stocks:
            await msg.edit_text(
                "⚠️ تعذّر جلب بيانات الأسهم. قد تكون البورصة مغلقة أو المصدر غير متاح."
            )
            return

        market_text = format_summary_text(market_summary) if market_summary else ""
        analysis_text = format_analysis_for_ai(stocks, market_text)
        ai_report = generate_arabic_report(analysis_text)

        main_message = build_telegram_message(ai_report, stocks, market_summary)
        await msg.delete()
        await update.message.reply_text(
            main_message,
            parse_mode=ParseMode.MARKDOWN,
        )

        stocks_table = build_stocks_table_message(stocks)
        if stocks_table and len(stocks_table) > 50:
            await update.message.reply_text(
                stocks_table,
                parse_mode=ParseMode.MARKDOWN,
            )

    except Exception as e:
        logger.error(f"Error in /report: {e}", exc_info=True)
        try:
            await update.message.reply_text(
                "❌ حدث خطأ أثناء تحليل الأسهم. يرجى المحاولة لاحقًا."
            )
        except Exception:
            pass


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick scan — just the top bullish/bearish stocks without AI summary."""
    msg = await update.message.reply_text("⏳ جاري مسح الأسهم…")

    try:
        stocks = scan_all_stocks()

        if not stocks:
            await msg.edit_text("⚠️ تعذّر جلب بيانات الأسهم.")
            return

        stocks_table = build_stocks_table_message(stocks)
        await msg.delete()
        await update.message.reply_text(
            stocks_table,
            parse_mode=ParseMode.MARKDOWN,
        )

    except Exception as e:
        logger.error(f"Error in /scan: {e}", exc_info=True)
        try:
            await update.message.reply_text(
                "❌ حدث خطأ أثناء المسح. يرجى المحاولة لاحقًا."
            )
        except Exception:
            pass


# ─── Scheduled Push (called by GitHub Actions) ───────────────────────────────

async def send_scheduled_report() -> None:
    """
    Push daily professional technical analysis report to TELEGRAM_CHAT_ID.
    Sends two messages: main report + detailed stocks table.
    """
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID is not set. Cannot send scheduled report.")
        sys.exit(1)

    bot = Bot(token=token)

    if not _is_egx_trading_day():
        logger.info("Today is not an EGX trading day (weekend). Skipping.")
        return

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⏳ جاري تحضير التقرير التقني الاحترافي…\n"
                "📊 يتم تحليل جميع أسهم البورصة المصرية\n"
                "🔬 حساب: RSI، Stochastic، MACD، Bollinger، Volume Profile، ADX، OBV، Williams %R"
            ),
        )

        market_summary = build_market_summary()
        stocks = scan_all_stocks()

        if not stocks:
            await bot.send_message(
                chat_id=chat_id,
                text="⚠️ تعذّر جلب بيانات الأسهم اليوم. قد تكون البورصة مغلقة أو المصدر غير متاح.",
            )
            logger.info("No stock data available. Sent notification and exiting.")
            return

        market_text = format_summary_text(market_summary) if market_summary else ""
        analysis_text = format_analysis_for_ai(stocks, market_text)
        ai_report = generate_arabic_report(analysis_text)

        main_message = build_telegram_message(ai_report, stocks, market_summary)
        await bot.send_message(
            chat_id=chat_id,
            text=main_message,
            parse_mode=ParseMode.MARKDOWN,
        )

        stocks_table = build_stocks_table_message(stocks)
        if stocks_table and len(stocks_table) > 50:
            await bot.send_message(
                chat_id=chat_id,
                text=stocks_table,
                parse_mode=ParseMode.MARKDOWN,
            )

        logger.info("Scheduled professional report sent successfully.")

    except Exception as e:
        # Sanitize error message — don't expose bot token in logs
        safe_err = str(e).replace(token, "[REDACTED]")
        logger.error(f"Failed to send scheduled report: {safe_err}", exc_info=True)
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ تعذّر إرسال التقرير التقني اليوم. يرجى التحقق من السجلات.",
            )
        except Exception:
            pass
        sys.exit(1)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    check_env()

    if "--scheduled" in sys.argv:
        import asyncio
        asyncio.run(send_scheduled_report())
        return

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("scan", cmd_scan))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
