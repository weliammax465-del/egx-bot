"""
ai_report.py
------------
AI Explanation Layer for EGX market analysis.

CRITICAL: AI ONLY explains already-computed results.
- AI never generates prices, scores, or technical values.
- AI never fabricates stock names or fundamentals.
- AI only summarizes and explains the computed data in Arabic.
- If data is missing or poor quality, AI says so honestly.
"""

import os
import re
import time
import logging
from datetime import datetime
import pytz
import google.generativeai as genai

from indicators import StockAnalysis

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

# Strict prompt: AI explains computed data, never generates numbers
SYSTEM_PROMPT = """أنت محلل مالي تقني متخصص في البورصة المصرية (EGX).

مهمتك الوحيدة هي شرح وتوضيح البيانات التقنية المحسوبة مسبقًا باللغة العربية.
كل الأرقام والمؤشرات والدرجات والتوصيات تم حسابها برمجيًا — دورك هو شرحها فقط.

قواعد صارمة:
- لا ت invent أو تخترع أي أرقام، أسعار، أو قيم مؤشرات.
- لا تذكر سعرًا أو درجة لم يرد في البيانات المقدمة لك.
- استخدم فقط الأرقام الموجودة في البيانات.
- اشرح سبب كل توصية بناءً على المؤشرات المذكورة.
- اذكر المخاطر بوضوح.
- استخدم لغة عربية واضحة ومهنية.
- لا تستخدم رموز Markdown.
- التقرير يجب أن يكون مناسبًا للقراءة على الهاتف.
- إذا كانت البيانات غير كافية، قل ذلك بصراحة.
- لا تقدم نصائح استثمارية شخصية.
- استخدم صيغة "الأسهم المرشحة" بدلاً من "أسهم ستصعد".
"""

_MARKDOWN_V1_SPECIAL = re.compile(r"([*_`\[])")


def _escape_markdown(text: str) -> str:
    """Escape only Markdown V1 special characters for Telegram."""
    if not text:
        return text
    return _MARKDOWN_V1_SPECIAL.sub(r"\\\1", text)


def _safe_truncate(text: str, max_len: int) -> str:
    """Truncate at a safe boundary (last newline before max_len)."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len]
    last_nl = truncated.rfind("\n")
    if last_nl > max_len - 200:
        return text[:last_nl] + "\n…"
    return truncated + "…"


def _format_arabic_date() -> str:
    """Return today's date in Arabic."""
    cairo_tz = pytz.timezone("Africa/Cairo")
    now = datetime.now(cairo_tz)
    day_ar = ARABIC_DAYS.get(now.strftime("%A"), "")
    month_ar = ARABIC_MONTHS.get(now.strftime("%B"), "")
    return f"{day_ar}، {now.day} {month_ar} {now.year}"


def explain_analysis(computed_data: str) -> str:
    """
    Use AI to EXPLAIN already-computed technical analysis data.
    AI does NOT generate any numbers — it only summarizes what's provided.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set.")

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
فيما يلي بيانات تقنية محسوبة لأسهم البورصة المصرية.
اكتب تقريرًا بالعربية يشرح هذه البيانات:

{computed_data}

التقرير يجب أن:
1. يبدأ بملخص قصير عن حالة السوق العامة
2. يشرح أبرز الأسهم المرشحة للصعود مع ذكر الأسباب التقنية
3. يحذر من الأسهم الهابطة
4. يذكر المخاطر بوضوح
5. يذكر أن هذه ليست نصيحة استثمارية

مهم: استخدم فقط الأرقام والبيانات المذكورة أعلاه. لا تخترع أي أرقام جديدة.
"""

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(user_prompt, request_options={"timeout": 90})
            if response.text:
                return response.text.strip()
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            err_str = str(e).lower()
            logger.error(f"Gemini error (attempt {attempt+1}): {e}")
            if "429" in err_str or "rate" in err_str or "quota" in err_str:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited. Waiting {wait}s…")
                time.sleep(wait)
                continue
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.warning("All Gemini attempts failed. Returning computed data as-is.")
    return computed_data[:2000]


# Backward-compatible alias
generate_arabic_report = explain_analysis


# ─── Telegram Message Builders ───────────────────────────────────────────────

def build_telegram_message(
    ai_summary: str,
    stocks: list[StockAnalysis],
    market_summary=None,
) -> str:
    """Assemble the full professional Telegram message with scoring data."""
    date_str = _format_arabic_date()

    lines = [
        "🇪🇬 *تقرير البورصة المصرية التقني*",
        f"📅 {date_str}",
        "",
    ]

    if market_summary:
        arrow = "📈" if market_summary.direction == "up" else ("📉" if market_summary.direction == "down" else "➡️")
        lines += [
            f"*مؤشر EGX 30:* {_escape_markdown(str(market_summary.current_value))} {arrow}",
            f"*التغيير:* {_escape_markdown(str(market_summary.change))} ({_escape_markdown(str(market_summary.change_pct))})",
        ]
        if market_summary.month_change_pct:
            lines.append(f"*شهري:* {_escape_markdown(str(market_summary.month_change_pct))}")
        if market_summary.year_change_pct:
            lines.append(f"*سنوي:* {_escape_markdown(str(market_summary.year_change_pct))}")
        lines.append("")

    # Score-based summary
    analyzed = [s for s in stocks if s.scoring_result is not None and s.data_quality >= 0.5]
    buy_count = sum(1 for s in analyzed if s.scoring_result.recommendation == "Buy")
    watch_count = sum(1 for s in analyzed if s.scoring_result.recommendation == "Watch")
    sell_count = sum(1 for s in analyzed if s.scoring_result.recommendation == "Sell")
    no_trade = len(analyzed) - buy_count - watch_count - sell_count

    lines += [
        f"📊 *الملخص:* {len(analyzed)} سهم محلل",
        f"🟢 شراء: {buy_count} | 🟡 مراقبة: {watch_count} | 🔴 بيع: {sell_count} | ⚪ لا تداول: {no_trade}",
        "",
        "─────────────────────",
        "",
        "🤖 *التحليل التقني:*",
        _escape_markdown(ai_summary),
        "",
    ]

    return _safe_truncate("\n".join(lines), 3800)


def build_stocks_table_message(stocks: list[StockAnalysis]) -> str:
    """
    Detailed stock table with v2 scores, signals, stop-loss, target, R/R ratio.
    Shows exclusion reasons for filtered-out stocks (analytics transparency).
    """
    lines = ["📊 *تفاصيل الأسهم — نظام Liquidity-First v2*", ""]

    # Separate by recommendation
    def _rec(s):
        return s.scoring_result.recommendation if s.scoring_result else "No Trade"

    top_buy   = [s for s in stocks if _rec(s) == "Buy"][:10]
    top_watch = [s for s in stocks if _rec(s) == "Watch"][:5]
    top_sell  = [s for s in stocks if _rec(s) == "Sell"][:5]
    excluded  = [s for s in stocks if _rec(s) == "No Trade"
                 and s.scoring_result and s.scoring_result.exclusion_reason][:8]

    if not top_buy and not top_watch and not top_sell:
        lines.append("⚪ لا توجد فرص تداول مؤكدة اليوم.")
        lines.append("لا يوجد سهم اجتاز فلاتر السيولة والتأكيد وإدارة المخاطر.")
        lines.append("")
        if excluded:
            lines.append(f"🚫 تم استبعاد {len(excluded)} سهم:")
            for s in excluded[:5]:
                excl = s.scoring_result.exclusion_reason
                lines.append(f"   • {_escape_markdown(s.ticker)}: {_escape_markdown(excl)}")
        lines.append("")
        lines.append("─────────────────────")
        return "\n".join(lines)

    if top_buy:
        lines.append("🟢 *شراء (درجة 70+):*")
        lines.append("")
        for i, s in enumerate(top_buy, 1):
            sr = s.scoring_result
            price_str = f"{s.current_price:.2f}"
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   💰 السعر: {price_str} EGP | 🎯 درجة: {sr.total_score}/100")
            # v2 risk management info
            if sr.stop_loss and sr.stop_loss > 0:
                lines.append(f"   🛑 وقف الخسارة: {sr.stop_loss:.2f} | 🎯 الهدف: {sr.target:.2f}")
                lines.append(f"   📐 نسبة R/R: {sr.rr_ratio:.1f}:1 | ⚠️ المخاطرة: {_escape_markdown(sr.risk_level)}")
            if sr.pass_reasons:
                lines.append(f"   ✅ {_escape_markdown(sr.pass_reasons[0])}")
            lines.append("")

    if top_watch:
        lines.append("🟡 *مراقبة (درجة 50-69):*")
        lines.append("")
        for i, s in enumerate(top_watch, 1):
            sr = s.scoring_result
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   💰 {s.current_price:.2f} EGP | 🎯 {sr.total_score}/100")
            if sr.stop_loss and sr.stop_loss > 0:
                lines.append(f"   🛑 وقف: {sr.stop_loss:.2f} | 🎯 هدف: {sr.target:.2f} | R/R: {sr.rr_ratio:.1f}:1")
            if sr.pass_reasons:
                lines.append(f"   👀 {_escape_markdown(sr.pass_reasons[0])}")
            lines.append("")

    if top_sell:
        lines.append("🔴 *بيع / تجنب (درجة 30 أو أقل):*")
        lines.append("")
        for i, s in enumerate(top_sell, 1):
            sr = s.scoring_result
            lines.append(f"{i}. *{_escape_markdown(s.name_ar)}* ({_escape_markdown(s.ticker)})")
            lines.append(f"   💰 {s.current_price:.2f} EGP | 🎯 {sr.total_score}/100")
            if sr.fail_reasons:
                lines.append(f"   ⚠️ {_escape_markdown(sr.fail_reasons[0])}")
            lines.append("")

    # Exclusion summary (analytics transparency — shows users WHY stocks were filtered)
    if excluded:
        lines.append("🚫 *مستبعدة من الفلاتر:*")
        lines.append("")
        for s in excluded:
            excl = s.scoring_result.exclusion_reason or "استبعاد تلقائي"
            lines.append(f"   • {_escape_markdown(s.ticker)}: {_escape_markdown(excl)}")
        lines.append("")

    lines += [
        "─────────────────────",
        "🔗 المصادر: TradingView | stockanalysis.com",
        "⏰ البيانات تقريبية — تأكد قبل اتخاذ أي قرار",
        "⚠️ هذه ليست نصيحة استثمارية",
    ]

    return _safe_truncate("\n".join(lines), 4000)


def format_stock_detail(stock: StockAnalysis) -> str:
    """
    Full single-stock analysis card for /stock command.
    Shows v2 score, recommendation, stop-loss, target, R/R, indicators.
    If excluded by a filter, shows the exclusion reason clearly.
    """
    change_sign = "+" if stock.daily_change_pct >= 0 else ""
    vol_str = f"{int(stock.volume):,}" if stock.volume > 0 else "غير متاح"

    lines = [
        f"📊 *{_escape_markdown(stock.name_ar)}* ({_escape_markdown(stock.ticker)})",
        "",
        f"💰 السعر: *{stock.current_price:.2f} EGP*",
        f"📈 التغيير: {change_sign}{stock.daily_change_pct:.2f}%",
        f"📦 الحجم: {vol_str}",
        "",
    ]

    if stock.scoring_result is not None:
        sr = stock.scoring_result

        # If stock was excluded by a filter — show that prominently
        if sr.exclusion_reason:
            lines += [
                f"🚫 *مستبعد من التوصيات*",
                f"   السبب: {_escape_markdown(sr.exclusion_reason)}",
                "",
                f"🎯 الدرجة: {sr.total_score}/100",
                f"🔄 البيانات: {_escape_markdown(sr.data_freshness)} (جودة: {sr.data_quality:.0%})",
                "",
            ]
        else:
            lines += [
                f"🎯 الدرجة: *{sr.total_score}/100*",
                f"📋 التوصية: {sr.recommendation_ar}",
                "",
            ]

            # v2 Risk Management block (only if computed)
            if sr.stop_loss and sr.stop_loss > 0:
                lines += [
                    "⚖️ *إدارة المخاطر (v2):*",
                    f"   🛑 وقف الخسارة: *{sr.stop_loss:.2f} EGP*",
                    f"   🎯 الهدف (مقاومة): *{sr.target:.2f} EGP*",
                    f"   📐 نسبة المكسب/الخسارة: *{sr.rr_ratio:.1f}:1*",
                    f"   ⚠️ المخاطرة: {_escape_markdown(sr.risk_level)} — {_escape_markdown(sr.risk_reason)}",
                    "",
                ]

            lines.append(f"🔄 البيانات: {_escape_markdown(sr.data_freshness)} (جودة: {sr.data_quality:.0%})")
            lines.append("")

            if sr.pass_reasons:
                lines.append("✅ *عوامل إيجابية:*")
                for r in sr.pass_reasons[:4]:
                    lines.append(f"   • {_escape_markdown(r)}")
                lines.append("")

            if sr.fail_reasons:
                lines.append("❌ *عوامل سلبية:*")
                for r in sr.fail_reasons[:4]:
                    lines.append(f"   • {_escape_markdown(r)}")
                lines.append("")

    # Support / Resistance from indicators
    if stock.support > 0 or stock.resistance > 0:
        lines.append("📌 *مستويات السعر:*")
        if stock.support > 0:
            lines.append(f"   دعم: {stock.support:.2f}")
        if stock.resistance > 0:
            lines.append(f"   مقاومة: {stock.resistance:.2f}")
        lines.append("")

    # Technical indicators
    if stock.indicators:
        lines.append("🔬 *المؤشرات التقنية:*")
        for ind in stock.indicators[:12]:
            lines.append(f"   {ind.name_ar}: {ind.value} ({ind.signal_text})")
        lines.append("")

    lines += [
        "─────────────────────",
        "🔗 TradingView + stockanalysis.com",
        "⚠️ للمعلومات فقط — ليست نصيحة استثمارية",
    ]

    return _safe_truncate("\n".join(lines), 4000)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from stock_scanner import scan_all_stocks, format_analysis_for_ai
    from fetch_egx import build_market_summary, format_summary_text

    print("Scanning stocks...")
    stocks = scan_all_stocks()
    print(f"Analyzed {len(stocks)} stocks")

    market = build_market_summary()
    market_text = format_summary_text(market) if market else ""
    full_text = market_text + "\n\n" + format_analysis_for_ai(stocks, market_text)

    print("Generating AI explanation...")
    report = explain_analysis(full_text)

    msg1 = build_telegram_message(report, stocks, market)
    msg2 = build_stocks_table_message(stocks)

    print(f"\n=== MAIN MESSAGE ({len(msg1)} chars) ===")
    print(msg1[:500])
    print(f"\n=== STOCKS TABLE ({len(msg2)} chars) ===")
    print(msg2[:500])
