"""
bot.py
------
Telegram bot entry point.

Commands:
  /start   — welcome message
  /report  — fetch and send today's EGX market report on demand

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
from ai_report import generate_arabic_report, build_telegram_message

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
        "🇪🇬 *مرحبًا بك في بوت البورصة المصرية\\!*\n\n"
        "يمكنك استخدام الأوامر التالية:\n"
        "• /report — تقرير السوق اليومي\n\n"
        "⚠️ _المعلومات للأغراض المعلوماتية فقط\\._",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetch market data, generate AI summary, send formatted report."""
    msg = await update.message.reply_text("⏳ جاري تحضير التقرير…")

    try:
        market_summary = build_market_summary()
        raw_text = format_summary_text(market_summary)
        arabic_report = generate_arabic_report(raw_text)
        full_message = build_telegram_message(arabic_report, market_summary)

        await msg.delete()
        await update.message.reply_text(
            full_message,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        logger.error(f"Error in /report: {e}")
        # Don't use msg.edit_text — msg might be deleted already
        try:
            await update.message.reply_text(
                "❌ حدث خطأ أثناء تحضير التقرير. يرجى المحاولة لاحقًا."
            )
        except Exception:
            pass


# ─── Scheduled Push (called by GitHub Actions) ───────────────────────────────

async def send_scheduled_report() -> None:
    """
    Push daily report to TELEGRAM_CHAT_ID.
    Called directly (not via polling) from GitHub Actions with --scheduled flag.
    """
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID is not set. Cannot send scheduled report.")
        sys.exit(1)

    bot = Bot(token=token)

    # Skip non-trading days to save API quota
    if not _is_egx_trading_day():
        logger.info("Today is not an EGX trading day (weekend). Skipping.")
        return

    try:
        market_summary = build_market_summary()

        # If market data is unavailable, notify but don't crash
        if not market_summary.is_trading_day:
            await bot.send_message(
                chat_id=chat_id,
                text="ℹ️ تعذّر جلب بيانات السوق اليوم. قد تكون البورصة مغلقة أو المصدر غير متاح.",
            )
            logger.info("Market data unavailable. Sent notification and exiting.")
            return

        raw_text = format_summary_text(market_summary)
        arabic_report = generate_arabic_report(raw_text)
        full_message = build_telegram_message(arabic_report, market_summary)

        await bot.send_message(
            chat_id=chat_id,
            text=full_message,
            parse_mode=ParseMode.MARKDOWN,
        )
        logger.info("Scheduled report sent successfully.")

    except Exception as e:
        logger.error(f"Failed to send scheduled report: {e}")
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="❌ تعذّر إرسال تقرير البورصة اليوم. يرجى التحقق من السجلات.",
            )
        except Exception:
            pass
        sys.exit(1)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    check_env()

    # If called with --scheduled flag, send once and exit (for GitHub Actions)
    if "--scheduled" in sys.argv:
        import asyncio
        asyncio.run(send_scheduled_report())
        return

    # Otherwise, start the interactive polling bot
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("report", cmd_report))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
