"""
ai_report.py
------------
Generates a short Arabic market summary using Google Gemini (free tier).
Requires GEMINI_API_KEY environment variable.

No financial advice is given. Output is informational only.
"""

import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-1.5-flash"  # Free tier, fast, supports Arabic

SYSTEM_PROMPT = """أنت محلل مالي مصري متخصص في البورصة المصرية (EGX).
مهمتك هي تقديم ملخص يومي موجز وواضح باللغة العربية عن حركة السوق.

القواعد الصارمة:
- لا تقدم نصائح استثمارية شخصية أبدًا.
- لا تضمن أرباحًا أو تتوقع مكاسب مستقبلية.
- أذكر دائمًا أن المعلومات للأغراض المعلوماتية فقط.
- استخدم لغة عربية واضحة وبسيطة.
- الملخص يجب أن يكون بين 5 إلى 8 جمل فقط.
- ابدأ بحالة المؤشر الرئيسي EGX 30، ثم أبرز الأسهم، ثم خاتمة تنبيهية.
"""


def generate_arabic_report(market_text: str) -> str:
    """
    Takes raw market data as English text and returns a concise Arabic report.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get your free key from https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    user_prompt = f"""
بناءً على البيانات التالية من البورصة المصرية ليوم اليوم، 
اكتب ملخصًا عربيًا موجزًا لا يزيد عن 8 جمل:

{market_text}

تذكّر: المعلومات للأغراض المعلوماتية فقط، ولا تمثل نصيحة استثمارية.
"""

    try:
        response = model.generate_content(user_prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return (
            "⚠️ تعذّر إنشاء الملخص الذكي في الوقت الحالي. "
            "يرجى المتابعة مع مزود الخدمة لاحقًا."
        )


def build_telegram_message(summary_ar: str, market_summary) -> str:
    """
    Assemble the full Telegram-formatted message with header, AI summary,
    raw index data, and disclaimer.
    """
    from datetime import datetime
    import pytz

    cairo_tz = pytz.timezone("Africa/Cairo")
    now = datetime.now(cairo_tz)
    date_str = now.strftime("%A، %d %B %Y")  # e.g. Sunday، 28 June 2026

    arrow = (
        "📈" if market_summary.direction == "up"
        else ("📉" if market_summary.direction == "down" else "➡️")
    )

    lines = [
        f"🇪🇬 *تقرير البورصة المصرية اليومي*",
        f"📅 {date_str}",
        "",
        f"*مؤشر EGX 30:* {market_summary.current_value} {arrow}",
        f"*التغيير:* {market_summary.change} ({market_summary.change_pct})",
        "",
        "─────────────────────",
        "",
        "🤖 *ملخص الذكاء الاصطناعي:*",
        summary_ar,
        "",
        "─────────────────────",
    ]

    # Append top gainers
    if market_summary.top_gainers:
        lines.append("📗 *أعلى الأسهم ارتفاعًا:*")
        for s in market_summary.top_gainers[:3]:
            lines.append(f"• {s['name']}: {s['price']} ({s['change_pct']})")
        lines.append("")

    # Append top losers
    if market_summary.top_losers:
        lines.append("📕 *أعلى الأسهم انخفاضًا:*")
        for s in market_summary.top_losers[:3]:
            lines.append(f"• {s['name']}: {s['price']} ({s['change_pct']})")
        lines.append("")

    # Disclaimer
    lines += [
        "─────────────────────",
        "⚠️ _هذا التقرير للأغراض المعلوماتية فقط ولا يمثل نصيحة استثمارية._",
        "_لا تتخذ قرارات استثمارية بناءً على هذه المعلومات وحدها._",
        "",
        f"🔗 المصدر: Investing.com | EGX",
    ]

    return "\n".join(lines)


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
