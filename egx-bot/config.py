"""
config.py
---------
Central configuration for EGX Bot — Liquidity-First Strategy v2.
All tunable parameters in one place for easy adjustment.
كل القيم القابلة للتعديل في مكان واحد.
"""

# ─── Liquidity Gate (فلتر السيولة) ────────────────────────────────────────────
# أقل قيمة تداول يومية مقبولة (بالجنيه المصري)
MIN_TURNOVER_EGP = 1_000_000
# يجب استمرار السيولة على هذا العدد من الأيام
MIN_TURNOVER_AVG_DAYS = 20

# ─── Price Limit Filter (فلتر حد التذبذب) ────────────────────────────────────
# EGX حد أقصى للتذبذب ±10% — سهم وصل للحد = مجمد، لا توصية عليه
PRICE_LIMIT_THRESHOLD_PCT = 10.0

# ─── Volume Surge (ارتفاع حجم التداول) ───────────────────────────────────────
# نافذة المقارنة لاكتشاف ارتفاع الحجم
VOLUME_SURGE_LOOKBACK_DAYS = 3
# تأكيد يومين متتاليين من ارتفاع الحجم
CONFIRMATION_DAYS = 2

# ─── Risk Management (إدارة المخاطر) ─────────────────────────────────────────
# لا تقبل أقل من نسبة ربح:خسارة 2:1
MIN_RISK_REWARD_RATIO = 2.0
# وقف الخسارة = 1.5 × ATR
ATR_STOP_LOSS_MULTIPLIER = 1.5

# ─── Momentum (الزخم) ────────────────────────────────────────────────────────
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0

# ─── Relative Strength (القوة النسبية) ───────────────────────────────────────
# مقارنة أداء السهم vs EGX 30 على هذا العدد من الأيام
RELATIVE_STRENGTH_PERIOD = 20

# ─── Scoring Thresholds (حدود التقييم) ───────────────────────────────────────
BUY_THRESHOLD = 70
WATCH_THRESHOLD = 50
SELL_THRESHOLD = 30

# ─── Data Quality (جودة البيانات) ────────────────────────────────────────────
MIN_DATA_QUALITY = 0.8
# فرق السعر بين المصدرين (stockanalysis vs tvDatafeed) — فوق 4% = مشبوه
PRICE_DEVIATION_THRESHOLD_PCT = 4.0
# بيانات أقدم من 24 ساعة = قديمة
STALE_DATA_HOURS = 24
# تغير يومي فوق 20% = خطأ في البيانات
SUSPICIOUS_PRICE_CHANGE_PCT = 20.0

# ─── Volume Surge Multiplier ─────────────────────────────────────────────────
# حجم التداول يجب أن يكون أكبر من المتوسط بهذا المضاعف ليعتبر surge
VOLUME_SURGE_MULTIPLIER = 2.0

# ─── Report Settings (إعدادات التقرير) ───────────────────────────────────────
MAX_BUY_RECOMMENDATIONS = 5
MAX_WATCH_RECOMMENDATIONS = 5
REPORT_TIMEZONE = "Africa/Cairo"
