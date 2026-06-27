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


def check_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)


# ─── Command Handlers ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🇪🇬 *مرحبًا بك في بوت البورصة المصرية!*\n\n"
        "يمكنك استخدام الأوامر التالية:\n"
        "• /report — تقرير السوق اليومي\n\n"
        "⚠️ _المعلومات للأغراض المعلوماتية فقط._",
        parse_mode=ParseMode.MARKDOWN,
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
        await msg.edit_text(
            "❌ حدث خطأ أثناء تحضير التقرير. يرجى المحاولة لاحقًا."
        )


# ─── Scheduled Push (called by GitHub Actions) ───────────────────────────────

async def send_scheduled_report() -> None:
    """
    Push daily report to TELEGRAM_CHAT_ID.
    Called directly (not via polling) from GitHub Actions.
    """
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not chat_id:
        logger.error("TELEGRAM_CHAT_ID is not set. Cannot send scheduled report.")
        sys.exit(1)

    bot = Bot(token=token)

    try:
        await bot.send_message(
            chat_id=chat_id,
            text="⏳ جاري تحضير تقرير البورصة اليومي…",
        )

        market_summary = build_market_summary()
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
