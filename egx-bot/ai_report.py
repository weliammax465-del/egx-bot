"""
ai_report.py
------------
Generates professional Arabic market analysis using Google Gemini (free tier).
Requires GEMINI_API_KEY environment variable.
"""

import os
import re
import time
import logging
from datetime import datetime
import pytz
import google.generativeai as genai

from indicators import StockAnalysis
from stock_scanner import get_top_bullish, get_top_bearish

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

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

SYSTEM_PROMPT = """أنت محلل مالي تقني محترف متخصص في البورصة المصرية (EGX).
خبرتك تشمل التحليل التقني المتقدم باستخدام:
- مؤشر القوة النسبية RSI
- مؤشر الاستوكاستك Stochastic
- ماكد MACD
- بولينجر باند Bollinger Bands
- المتوسطات المتحركة SMA (20، 50، 200)
- مؤشر الاتجاه ADX
- مؤشر التوازن الحجمي OBV
- ويليامز %R
- فوليوم بروفايل Volume Profile
- متوسط المدى الحقيقي ATR

مهمتك:
1. تحليل البيانات التقنية لأسهم البورصة المصرية
2. تحديد الأسهم الأكثر احتمالية للصعود بناءً على تقاطع المؤشرات
3. تقديم تبرير تقني واضح لكل توصية
4. تحديد مستويات الدعم والمقاومة

القواعد الصارمة:
- لا تقدم نصائح استثمارية شخصية أبدًا.
- لا تضمن أرباحًا أو تتوقع نتائج مؤكدة.
- استخدم صيغة "الأسهم المرشحة للصعود" بدلاً من "أسهم ستصعد".
- استخدم لغة عربية واضحة ومهنية.
- لا تستخدم رموز Markdown مثل * أو _ أو [ في النص العادي.
- التقرير يجب أن يكون منظمًا ومناسبًا للقراءة على الهاتف.
- لكل سهم مرشح للصعود: اذكر اسم السهم، السعر الحالي، السبب التقني الرئيسي، ومستوى الدعم/المقاومة.
- أضف قسمًا للأسهم الهابطة (تحذير).
"""

# Telegram Markdown V1 — only these chars need escaping
_MARKDOWN_V1_SPECIAL = re.compile(r"([*_`\[])")


def _escape_markdown(text: str) -> str:
    """Escape only Markdown V1 special characters for Telegram."""
    if not text:
        return text
    return _MARKDOWN_V1_SPECIAL.sub(r"\\\1", text)


def _safe_truncate(text: str, max_len: int) -> str:
    """Truncate at a safe boundary (last newline before max_len) to avoid breaking markdown."""
    if len(text) <= max_len:
        return text
    # Find the last newline before max_len
    truncated = text[:max_len]
    last_nl = truncated.rfind("\n")
    if last_nl > max_len - 200:  # Only use newline if it's reasonably close
        return text[:last_nl] + "\n…"
    return truncated + "…"


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
    Takes raw market/stock data as text and returns a professional Arabic report.
    Includes retry logic with 429 rate-limit handling and graceful fallback.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get your free key from https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)

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
بناءً على البيانات التقنية التالية لأسهم البورصة المصرية،
اكتب تقريرًا تحليليًا احترافيًا باللغة العربية:

{market_text}

التقرير يجب أن يحتوي على:
1. مقدمة موجزة عن حالة السوق العامة
2. قائمة بالأسهم المرشحة للصعود (أعلى 5-10 أسهم) مع التبرير التقني لكل سهم
3. قائمة بالأسهم المرشحة للهبوط (أعلى 3-5 أسهم، تحذير) مع السبب

اجعل التقرير مختصرًا ومناسبًا للقراءة على الهاتف المحمول.
"""

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                user_prompt,
                request_options={"timeout": 90},
            )
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    continue
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"Gemini API error (attempt {attempt + 1}/{max_retries + 1}): {e}")
            # 429 rate limit — wait longer
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s before retry…")
                time.sleep(wait)
                continue
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue

    logger.warning("All Gemini attempts failed. Returning fallback summary.")
    return "⚠️ تعذّر إنشاء التحليل الذكي في الوقت الحالي.\n\n" + market_text[:2000]


def build_telegram_message(
    ai_summary: str,
    stocks: list[StockAnalysis],
    market_summary=None,
) -> str:
    """
    Assemble the full professional Telegram-formatted message.
    Handles Telegram's 4096 character limit with safe truncation.
    """
    date_str = _format_arabic_date()

    lines = [
        "🇪🇬 *تقرير البورصة المصرية التقني الاحترافي*",
        f"📅 {date_str}",
        "",
    ]

    if market_summary:
        arrow = (
            "📈" if market_summary.direction == "up"
            else ("📉" if market_summary.direction == "down" else "➡️")
        )
        lines += [
            f"*مؤشر EGX 30:* {_escape_markdown(str(market_summary.current_value))} {arrow}",
            f"*التغيير اليومي:* {_escape_markdown(str(market_summary.change))} "
            f"({_escape_markdown(str(market_summary.change_pct))})",
        ]
        if market_summary.month_change_pct:
            lines.append(f"*الأداء الشهري:* {_escape_markdown(str(market_summary.month_change_pct))}")
        if market_summary.year_change_pct:
            lines.append(f"*الأداء السنوي:* {_escape_markdown(str(market_summary.year_change_pct))}")
        lines.append("")

    total_stocks = len(stocks)
    bullish_count = sum(1 for s in stocks if s.composite_score >= 2)
    bearish_count = sum(1 for s in stocks if s.composite_score <= -2)
    neutral_count = total_stocks - bullish_count - bearish_count

    lines += [
        f"📊 *ملخص المسح التقني:* {total_stocks} سهم",
        f"🟢 صاعدة: {bullish_count} | 🔴 هابطة: {bearish_count} | 🟡 محايدة: {neutral_count}",
        "",
        "─────────────────────",
        "",
        "🤖 *التحليل التقني الاحترافي:*",
        _escape_markdown(ai_summary),
        "",
    ]

    message = "\n".join(lines)
    message = _safe_truncate(message, 3800)

    if len(message) > 3800:
        logger.info("Main message truncated to fit Telegram limit.")

    return message


def build_stocks_table_message(stocks: list[StockAnalysis]) -> str:
    """
    Build a separate message with detailed stock tables (top bullish + bearish).
    Sent as a second message after the main report.
    """
    lines = ["📊 *تفاصيل الأسهم المرشحة*", ""]

    top_bull = get_top_bullish(stocks, 10)
    top_bear = get_top_bearish(stocks, 5)

    if top_bull:
        lines.append("🟢 *الأسهم المرشحة للصعود:*")
        lines.append("")
        for i, s in enumerate(top_bull, 1):
            ticker_clean = s.ticker.replace(".CA", "")
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(ticker_clean)})")
            lines.append(
                f"   السعر: {_escape_markdown(str(s.current_price))} "
                f"| التغيير: {_escape_markdown(str(s.daily_change_pct))}%"
            )
            lines.append(
                f"   الإشارة: {_escape_markdown(s.signal_label)} "
                f"| الثقة: {_escape_markdown(str(s.signal_score_pct))}%"
            )
            if s.bullish_reasons:
                for reason in s.bullish_reasons[:3]:
                    lines.append(f"   ✅ {_escape_markdown(reason)}")
            lines.append("")

    if top_bear:
        lines.append("🔴 *الأسهم المرشحة للهبوط (تحذير):*")
        lines.append("")
        for i, s in enumerate(top_bear, 1):
            ticker_clean = s.ticker.replace(".CA", "")
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(ticker_clean)})")
            lines.append(
                f"   السعر: {_escape_markdown(str(s.current_price))} "
                f"| الإشارة: {_escape_markdown(s.signal_label)}"
            )
            if s.bearish_reasons:
                for reason in s.bearish_reasons[:2]:
                    lines.append(f"   ⚠️ {_escape_markdown(reason)}")
            lines.append("")

    lines += [
        "─────────────────────",
        "🔗 المصادر: TradingView | stockanalysis.com",
    ]

    message = "\n".join(lines)
    message = _safe_truncate(message, 4000)

    return message


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from stock_scanner import scan_all_stocks, format_analysis_for_ai
    from fetch_egx import build_market_summary, format_summary_text

    print("Scanning stocks...")
    stocks = scan_all_stocks()
    print(f"Analyzed {len(stocks)} stocks")

    market = build_market_summary()
    market_text = format_summary_text(market)
    full_text = market_text + "\n\n" + format_analysis_for_ai(stocks)

    print("Generating AI report...")
    report = generate_arabic_report(full_text)

    msg1 = build_telegram_message(report, stocks, market)
    msg2 = build_stocks_table_message(stocks)

    print("\n=== MAIN MESSAGE ===")
    print(msg1)
    print(f"\n({len(msg1)} chars)")
    print("\n=== STOCKS TABLE ===")
    print(msg2)
    print(f"\n({len(msg2)} chars)")
