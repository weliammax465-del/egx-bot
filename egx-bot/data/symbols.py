"""
data/symbols.py
---------------
Symbol validation and mapping for EGX-listed instruments.

Ensures every ticker is a real EGX-listed stock.
Prevents hallucinated or invalid symbols from entering the pipeline.
"""

from __future__ import annotations

import json
import os
import logging

logger = logging.getLogger(__name__)

# ─── Arabic Names for EGX Stocks ────────────────────────────────────────────

ARABIC_NAMES: dict[str, str] = {
    "COMI": "البنك التجاري الدولي", "TMGH": "طلعت مصطفى", "SWDY": "السويدي إليكتريك",
    "ETEL": "المصرية للاتصالات", "EGAL": "مصر للألمنيوم", "EAST": "الشرقية للدخان",
    "QNBE": "بنك قطر الوطني", "MFPC": "مصر للأسمدة", "ABUK": "أبو قير للأسمدة",
    "HDBK": "بنك الإسكان والتعمير", "ALCN": "الإسكندرية للحاويات", "ORAS": "أوراسكوم للإنشاء",
    "EFIH": "إي فاينانس", "ADIB": "بنك أبوظبي الإسلامي", "EMFD": "عمار مصر",
    "FWRY": "فوري", "SCTS": "قناة السويس للتكنولوجيا", "ORHD": "أوراسكوم للتنمية",
    "PHDC": "بالم هيلز", "GPPL": "الأهرام القابضة", "VLMR": "فالمور",
    "VLMRA": "فالمور (أ)", "HRHO": "إي إف جي هيرميس", "EFID": "إديتا",
    "JUFO": "جهينة", "CANA": "بنك قناة السويس", "GBCO": "جي بي كورب",
    "OCDI": "سوديك", "BTFH": "بلتون", "RAYA": "رايا القابضة",
    "IRON": "ال حديد والصلب", "FERC": "فيركيم", "CIEB": "كريدي أجريكول",
    "FAIT": "بنك فيصل الإسلامي", "FAITA": "بنك فيصل الإسلامي (أ)",
    "HELI": "مدينة هليوبوليس", "EGCH": "الكيماويات المصرية", "VALU": "فاليو",
    "EXPA": "بنك التنمية الصادرات", "CLHO": "مستشفيات كليوباترا",
    "ARCC": "أسمنت العربية", "CCAP": "قالا", "TAQA": "طاقة",
    "EFIC": "الصناعات المالية والصناعية", "POUL": "دواجن القاهرة",
    "SKPC": "سيدي كرير", "EGTS": "المنتجعات المصرية", "MTIE": "مجموعة أم أم",
    "CIRA": "سيرا للتعليم", "SCEM": "أسمنت سيناء", "EGSA": "نايل سات",
    "MCQE": "أسمنت قنا", "SAUD": "بنك البركة", "ORWE": "نسج الشرق",
    "MASR": "مدينة مصر", "PHAR": "إيبيكو", "UBEE": "البنك المتحد",
    "MHOT": "فنادق مصر", "MBSC": "أسمنت بني سويف", "ISPH": "ابن سينا فارما",
    "CICH": "سي آي كابيتال", "EGBE": "بنك الخليج", "TALM": "تعليم",
    "ATQA": "عتاقة للصلب", "MOIL": "ماريدايف", "BINV": "استثمارات ب",
    "RMDA": "راميدا", "AMOC": "العامة للبترول", "IFAP": "المحاصيل الزراعية",
    "CSAG": "وكالات الشحن بالقناة", "OLFI": "أوبور لاند", "ISMQ": "مناجم الحديد",
    "BONY": "بنيان", "NIPH": "نايل فارما", "SPHT": "شمس للسياحة",
    "DOMT": "دومتي", "MIPH": "مينافارم", "KORA": "كورة للطاقة",
    "OIH": "أوراسكوم للاستثمار", "PRDC": "رواد العقارية", "MPRC": "مدينة الإنتاج",
    "EGAS": "غاز مصر", "ELEC": "كابلات مصر", "SUGR": "سكر الدلتا",
    "ZMID": "الزهراء المعادي", "ACAP": "أيه كابيتال", "AMES": "مركز الإسكندرية الطبي",
    "MOIN": "مهندس للتأمين", "BIOC": "جلاكسو", "PHTV": "بيراميزا",
    "NAPR": "الطباعة الوطنية", "CNFN": "كونتاكت", "CPCI": "كحيرة للأدوية",
    "AXPH": "الإسكندرية للأدوية", "NINH": "مستشفى النزهة", "MPCI": "ممفيس للأدوية",
    "ENGC": "آيكون", "GOUR": "جورميه", "SPIN": "الغزل والنسيج",
    "DSCW": "دايس", "MFSC": "محلات مصر الحرة", "SVCE": "أسمنت وادي النيل",
    "AMIA": "عرب ملتقى", "GSSC": "الصوامع العامة", "GDWA": "جدوة",
    "OCPH": "أكتوبر فارما", "MICH": "الكيماويات المصرية صناعات",
    "WCDF": "مطاحن الدلتا", "AJWA": "أجوة", "KABO": "كابو",
    "SAIB": "البنك العربي الأفريقي", "UEFM": "مطاحن صعيد مصر",
    "ACTF": "أكت فايننشال", "UNIT": "المتحدة للإسكان", "ASCM": "أسكوم",
    "ADCI": "الدواء العربي", "ARAB": "العربية للتعمير", "OFH": "أو بي",
    "ACAMD": "إدارة الأصول العربية", "ISMA": "إسماعيلية للدواجن",
    "ELSH": "الشمس للإسكان", "ETRS": "إيجي ترانس", "SDTI": "شرم دريمز",
    "KZPC": "كفر الزيات", "ACGC": "القطن العربية", "LCSW": "ليسيكو",
    "CFGH": "كونكريت فاشون", "ALRA": "أطلس", "ELKA": "القاهرة للإسكان",
    "AFMC": "مطاحن الإسكندرية", "ZEOT": "الزيوت", "AMER": "أمير جروب",
    "ATLC": "التوفيق للتأجير", "PHGC": "بريميوم هيلث",
    "SNFC": "أمن الغذاء بالشرقية", "EDFM": "مطاحن الدلتا الشرقية",
    "NAHO": "نعيم", "GGRN": "جو جرين", "DAPH": "دي إف جي",
    "INFI": "إسماعيلية للصناعات الغذائية",
}

# Path to the fallback stock list (canonical EGX symbols)
_FALLBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "egx_stocks.json")

# Cache of canonical symbols — loaded once from egx_stocks.json
_canonical_symbols: set[str] | None = None
_canonical_names: dict[str, str] = {}


def _load_canonical_list() -> None:
    """Load the canonical EGX symbol list from the fallback file."""
    global _canonical_symbols, _canonical_names
    if _canonical_symbols is not None:
        return
    try:
        with open(_FALLBACK_FILE, "r", encoding="utf-8") as f:
            stocks = json.load(f)
        _canonical_symbols = {s["symbol"] for s in stocks if "symbol" in s}
        for s in stocks:
            if "symbol" in s and "name" in s:
                _canonical_names[s["symbol"]] = s["name"]
        logger.info(f"Loaded {len(_canonical_symbols)} canonical EGX symbols.")
    except Exception as e:
        logger.error(f"Failed to load canonical symbol list: {e}")
        _canonical_symbols = set()
        _canonical_names = {}


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a stock symbol: remove .CA suffix, trim whitespace, uppercase.
    """
    if not symbol:
        return ""
    return symbol.strip().upper().replace(".CA", "").replace(".EG", "")


def is_valid_egx_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a verified EGX-listed instrument.
    Uses the canonical list from egx_stocks.json as source of truth.
    """
    _load_canonical_list()
    normalized = normalize_symbol(symbol)
    if not normalized:
        return False
    # Only accept symbols in the canonical EGX list
    if normalized in _canonical_symbols:
        return True
    logger.debug(f"Symbol {normalized} not found in canonical EGX list.")
    return False


def get_arabic_name(symbol: str, fallback_name: str = "") -> str:
    """
    Get the Arabic name for a symbol.
    Falls back to the English name if no Arabic name is available.
    """
    normalized = normalize_symbol(symbol)
    if normalized in ARABIC_NAMES:
        return ARABIC_NAMES[normalized]
    return fallback_name or normalized


def get_canonical_name(symbol: str) -> str:
    """Get the English name from the canonical list."""
    _load_canonical_list()
    normalized = normalize_symbol(symbol)
    return _canonical_names.get(normalized, "")


def update_canonical_list(stocks: list[dict]) -> None:
    """
    Update the canonical symbol set with freshly scraped data.
    Only adds symbols that already exist in the canonical list — does NOT accept
    unknown symbols from external sources (security: prevents injection of fake symbols).
    """
    global _canonical_symbols, _canonical_names
    _load_canonical_list()
    for s in stocks:
        sym = normalize_symbol(s.get("symbol", ""))
        if sym and sym in _canonical_symbols:  # Only update names for known symbols
            if "name" in s and s["name"]:
                _canonical_names[sym] = s["name"]


def validate_stock_entry(symbol: str, name: str, price: float) -> bool:
    """
    Validate a scraped stock entry.
    Returns True only if the symbol is known AND the price is positive.
    """
    if not is_valid_egx_symbol(symbol):
        logger.warning(f"Rejected unknown symbol: {symbol}")
        return False
    if not name or not name.strip():
        logger.warning(f"Rejected {symbol}: empty name")
        return False
    if price <= 0:
        logger.warning(f"Rejected {symbol}: invalid price {price}")
        return False
    return True
