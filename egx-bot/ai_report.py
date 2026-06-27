"""
ai_report.py
------------
Generates a short Arabic market summary using Google Gemini (free tier).
Requires GEMINI_API_KEY environment variable.

No financial advice is given. Output is informational only.
"""

import os
import re
import time
import logging
from datetime import datetime
import pytz
import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"  # Free tier, fast, supports Arabic

# Arabic day/month names
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

SYSTEM_PROMPT = """أنت محلل مالي مصري متخصص في البورصة المصرية (EGX).
مهمتك هي تقديم ملخص يومي موجز وواضح باللغة العربية عن حركة السوق.

القواعد الصارمة:
- لا تقدم نصائح استثمارية شخصية أبدًا.
- لا تضمن أرباحًا أو تتوقع مكاسب مستقبلية.
- أذكر دائمًا أن المعلومات للأغراض المعلوماتية فقط.
- استخدم لغة عربية واضحة وبسيطة.
- الملخص يجب أن يكون بين 5 إلى 8 جمل فقط.
- ابدأ بحالة المؤشر الرئيسي EGX 30، ثم الأداء الشهري والسنوي، ثم خاتمة تنبيهية.
- لا تستخدم رموز Markdown مثل * أو _ أو [ في النص.
"""

# Telegram Markdown special chars that need escaping
MARKDOWN_SPECIAL = re.compile(r"([_*\[\]`(){}~#>!\-])")


def _escape_markdown(text: str) -> str:
    """Escape Markdown special characters for Telegram."""
    if not text:
        return text
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _format_arabic_date() -> str:
    """Return today's date in Arabic (e.g. 'الأحد، 28 يونيو 2026')."""
    cairo_tz = pytz.timezone("Africa/Cairo")
    now = datetime.now(cairo_tz)
    day_en = now.strftime("%A")
    month_en = now.strftime("%B")
    day_num = now.day
    year = now.year

    day_ar = ARABIC_DAYS.get(day_en, day_en)
    month_ar = ARABIC_MONTHS.get(month_en, month_en)

    return f"{day_ar}، {day_num} {month_ar} {year}"


def generate_arabic_report(market_text: str) -> str:
    """
    Takes raw market data as English text and returns a concise Arabic report.
    Includes retry logic and graceful fallback.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get your free key from https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)

    # Configure safety settings — allow financial discussion
    safety_settings = {
        "harassment": "block_none",
        "hate_speech": "block_none",
        "sexually_explicit": "block_none",
        "dangerous": "block_only_high",
    }

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        safety_settings=safety_settings,
    )

    user_prompt = f"""
بناءً على البيانات التالية من البورصة المصرية ليوم اليوم،
اكتب ملخصًا عربيًا موجزًا لا يزيد عن 8 جمل:

{market_text}

تذكّر: المعلومات للأغراض المعلوماتية فقط، ولا تمثل نصيحة استثمارية.
"""

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                user_prompt,
                request_options={"timeout": 30},  # 30 second timeout
            )
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    # Fallback — return raw data summary without AI
    logger.warning("All Gemini attempts failed. Returning fallback summary.")
    return (
        "⚠️ تعذّر إنشاء الملخص الذكي في الوقت الحالي. "
        "إليك البيانات الخام:\n\n" + market_text
    )


def build_telegram_message(summary_ar: str, market_summary) -> str:
    """
    Assemble the full Telegram-formatted message with header, AI summary,
    raw index data, performance stats, and disclaimer.
    Handles Telegram's 4096 character limit.
    """
    date_str = _format_arabic_date()

    arrow = (
        "📈" if market_summary.direction == "up"
        else ("📉" if market_summary.direction == "down" else "➡️")
    )

    lines = [
        "🇪🇬 *تقرير البورصة المصرية اليومي*",
        f"📅 {date_str}",
        "",
        f"*مؤشر EGX 30:* {_escape_markdown(str(market_summary.current_value))} {arrow}",
        f"*التغيير اليومي:* {_escape_markdown(str(market_summary.change))} "
        f"({_escape_markdown(str(market_summary.change_pct))})",
    ]

    # Add monthly and yearly performance if available
    if market_summary.month_change_pct:
        lines.append(
            f"*الأداء الشهري:* {_escape_markdown(str(market_summary.month_change_pct))}"
        )
    if market_summary.year_change_pct:
        lines.append(
            f"*الأداء السنوي:* {_escape_markdown(str(market_summary.year_change_pct))}"
        )

    lines += [
        "",
        "─────────────────────",
        "",
        "🤖 *ملخص الذكاء الاصطناعي:*",
        _escape_markdown(summary_ar),
        "",
        "─────────────────────",
    ]

    # Append top gainers if available
    if market_summary.top_gainers:
        lines.append("📗 *أعلى الأسهم ارتفاعًا:*")
        for s in market_summary.top_gainers[:3]:
            name = _escape_markdown(str(s.get("name", "—")))
            price = _escape_markdown(str(s.get("price", "—")))
            pct = _escape_markdown(str(s.get("change_pct", "—")))
            lines.append(f"• {name}: {price} ({pct})")
        lines.append("")

    # Append top losers if available
    if market_summary.top_losers:
        lines.append("📕 *أعلى الأسهم انخفاضًا:*")
        for s in market_summary.top_losers[:3]:
            name = _escape_markdown(str(s.get("name", "—")))
            price = _escape_markdown(str(s.get("price", "—")))
            pct = _escape_markdown(str(s.get("change_pct", "—")))
            lines.append(f"• {name}: {price} ({pct})")
        lines.append("")

    # Disclaimer
    lines += [
        "─────────────────────",
        "⚠️ _هذا التقرير للأغراض المعلوماتية فقط ولا يمثل نصيحة استثمارية._",
        "_لا تتخذ قرارات استثمارية بناءً على هذه المعلومات وحدها._",
        "",
        "🔗 المصدر: Trading Economics | EGX",
    ]

    message = "\n".join(lines)

    # Telegram message limit is 4096 chars
    if len(message) > 4096:
        message = message[:4090] + "\n…\n"
        logger.warning("Telegram message was truncated to fit 4096 char limit.")

    return message


if __name__ == "__main__":
    # Quick local test with mock data
    from fetch_egx import build_market_summary, format_summary_text

    summary = build_market_summary()
    text = format_summary_text(summary)
    print("=== Raw Market Data ===")
    print(text)
    print("\n=== Arabic AI Report ===")
    report = generate_arabic_report(text)
    print(report)
    print("\n=== Full Telegram Message ===")
    print(build_telegram_message(report, summary))
