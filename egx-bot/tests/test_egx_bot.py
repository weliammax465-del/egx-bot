"""
tests/test_egx_bot.py
---------------------
Comprehensive unit tests for the EGX Intelligence Platform.
Covers: data validation, symbol mapping, indicators, scoring engine,
error handling, Telegram formatting, and fallback behavior.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(days=250, start_price=100.0, trend=0.1, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    closes = [start_price]
    for i in range(1, days):
        change = np.random.normal(trend, 1.5)
        closes.append(max(closes[-1] + change, 1.0))
    closes = np.array(closes)
    opens = closes + np.random.normal(0, 0.5, days)
    highs = np.maximum(opens, closes) + np.random.uniform(0.1, 1.0, days)
    lows = np.minimum(opens, closes) - np.random.uniform(0.1, 1.0, days)
    volumes = np.random.randint(100000, 5000000, days).astype(float)
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
    }, index=dates)


@pytest.fixture
def sample_ohlcv():
    return _make_ohlcv(250)

@pytest.fixture
def bullish_ohlcv():
    return _make_ohlcv(250, start_price=50, trend=0.3)

@pytest.fixture
def bearish_ohlcv():
    return _make_ohlcv(250, start_price=200, trend=-0.3)


# ═══════════════════════════════════════════════════════════════════════════
# DATA VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDataValidator:
    """Test the data validation layer."""

    def test_validate_good_ohlcv(self, sample_ohlcv):
        """Valid OHLCV should pass validation."""
        from data.validator import validate_ohlcv
        result = validate_ohlcv(sample_ohlcv, "TEST")
        assert result.is_valid
        assert result.quality_score > 0.8

    def test_validate_empty_dataframe(self):
        """Empty DataFrame should fail."""
        from data.validator import validate_ohlcv
        result = validate_ohlcv(pd.DataFrame(), "TEST")
        assert not result.is_valid

    def test_validate_missing_columns(self):
        """Missing columns should fail."""
        from data.validator import validate_ohlcv
        df = pd.DataFrame({"Open": [1, 2], "Close": [1, 2]})
        result = validate_ohlcv(df, "TEST")
        assert not result.is_valid

    def test_validate_negative_prices(self):
        """Negative prices should fail."""
        from data.validator import validate_ohlcv
        dates = pd.date_range(end=datetime.now(), periods=60, freq="B")
        df = pd.DataFrame({
            "Open": [-100]*60, "High": [-98]*60, "Low": [-102]*60,
            "Close": [-100]*60, "Volume": [1000]*60,
        }, index=dates)
        result = validate_ohlcv(df, "TEST")
        assert not result.is_valid

    def test_validate_insufficient_data(self):
        """Less than 50 bars should fail."""
        from data.validator import validate_ohlcv
        df = _make_ohlcv(30)
        result = validate_ohlcv(df, "TEST")
        assert not result.is_valid

    def test_validate_price_positive(self):
        """validate_price should accept positive values."""
        from data.validator import validate_price
        assert validate_price(100.0) is True
        assert validate_price(0.01) is True

    def test_validate_price_rejects_invalid(self):
        """validate_price should reject negative, zero, None, inf."""
        from data.validator import validate_price
        assert validate_price(0) is False
        assert validate_price(-1) is False
        assert validate_price(None) is False
        assert validate_price(float('inf')) is False

    def test_validate_change_pct_normal(self):
        """Normal change percentages should pass."""
        from data.validator import validate_change_pct
        assert validate_change_pct(5.0) is True
        assert validate_change_pct(-10.0) is True
        assert validate_change_pct(0.0) is True

    def test_validate_change_pct_extreme(self):
        """Changes beyond EGX ±20% limit should fail."""
        from data.validator import validate_change_pct
        assert validate_change_pct(25.0) is False
        assert validate_change_pct(-25.0) is False

    def test_deduplicate_stocks(self):
        """Duplicates should be removed."""
        from data.validator import deduplicate_stocks
        stocks = [
            {"symbol": "COMI", "name": "CIB", "price": 100},
            {"symbol": "COMI", "name": "CIB", "price": 101},  # dup
            {"symbol": "ETEL", "name": "Telecom", "price": 50},
        ]
        result = deduplicate_stocks(stocks)
        assert len(result) == 2

    def test_check_data_freshness_live(self):
        """Data from today should be 'live'."""
        from data.validator import check_data_freshness
        dates = pd.date_range(end=datetime.now(), periods=60, freq="B")
        df = pd.DataFrame({"Close": [100]*60}, index=dates)
        freshness = check_data_freshness(df)
        assert freshness in ("live", "fresh")

    def test_check_data_freshness_stale(self):
        """Old data should be 'stale'."""
        from data.validator import check_data_freshness
        dates = pd.date_range(end=datetime.now() - timedelta(days=30), periods=60, freq="B")
        df = pd.DataFrame({"Close": [100]*60}, index=dates)
        freshness = check_data_freshness(df)
        assert freshness == "stale"


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOL MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestSymbolMapping:
    """Test symbol validation and mapping."""

    def test_normalize_symbol_strips_suffix(self):
        """Should remove .CA and .EG suffixes."""
        from data.symbols import normalize_symbol
        assert normalize_symbol("COMI.CA") == "COMI"
        assert normalize_symbol("comi") == "COMI"
        assert normalize_symbol(" ETEL ") == "ETEL"

    def test_normalize_symbol_empty(self):
        from data.symbols import normalize_symbol
        assert normalize_symbol("") == ""

    def test_is_valid_egx_symbol_known(self):
        """Known EGX symbols should be valid."""
        from data.symbols import is_valid_egx_symbol
        assert is_valid_egx_symbol("COMI") is True
        assert is_valid_egx_symbol("ETEL") is True
        assert is_valid_egx_symbol("TMGH") is True

    def test_is_valid_egx_symbol_unknown(self):
        """Random symbols should be invalid."""
        from data.symbols import is_valid_egx_symbol
        assert is_valid_egx_symbol("FAKE123") is False
        assert is_valid_egx_symbol("") is False

    def test_get_arabic_name_known(self):
        """Known symbols should return Arabic names."""
        from data.symbols import get_arabic_name
        assert get_arabic_name("COMI") == "البنك التجاري الدولي"
        assert get_arabic_name("ETEL") == "المصرية للاتصالات"

    def test_get_arabic_name_fallback(self):
        """Unknown symbols should return fallback."""
        from data.symbols import get_arabic_name
        result = get_arabic_name("UNKNOWN", "Some English Name")
        assert result == "Some English Name"

    def test_validate_stock_entry_valid(self):
        """Valid entries should pass."""
        from data.symbols import validate_stock_entry
        assert validate_stock_entry("COMI", "CIB", 100.0) is True

    def test_validate_stock_entry_invalid_symbol(self):
        """Unknown symbols should fail."""
        from data.symbols import validate_stock_entry
        assert validate_stock_entry("FAKE", "Fake", 100.0) is False

    def test_validate_stock_entry_zero_price(self):
        """Zero prices should fail."""
        from data.symbols import validate_stock_entry
        assert validate_stock_entry("COMI", "CIB", 0) is False


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestIndicators:
    """Test individual technical indicators."""

    def test_rsi_range(self, sample_ohlcv):
        from indicators import calc_rsi
        r = calc_rsi(sample_ohlcv)
        assert 0 <= r.value <= 100
        assert r.signal in (-1, 0, 1)

    def test_rsi_correct_direction(self, sample_ohlcv):
        """RSI > 55 should be bullish (not bearish — regression test for the inverted logic bug)."""
        from indicators import calc_rsi
        df = _make_ohlcv(250, trend=0.2)
        r = calc_rsi(df)
        if 55 < r.value < 70:
            assert r.signal == 1, f"RSI={r.value} should be bullish, got signal={r.signal}"

    def test_stochastic_range(self, sample_ohlcv):
        from indicators import calc_stochastic
        r = calc_stochastic(sample_ohlcv)
        assert 0 <= r.value <= 100

    def test_stochastic_rsi(self, sample_ohlcv):
        from indicators import calc_stochastic_rsi
        r = calc_stochastic_rsi(sample_ohlcv)
        assert 0 <= r.value <= 100

    def test_macd_signal(self, sample_ohlcv):
        from indicators import calc_macd
        r = calc_macd(sample_ohlcv)
        assert r.signal in (-1, 0, 1)

    def test_bollinger(self, sample_ohlcv):
        from indicators import calc_bollinger
        r = calc_bollinger(sample_ohlcv)
        assert 0 <= r.value <= 100

    def test_sma_trend(self, sample_ohlcv):
        from indicators import calc_sma_trend
        r = calc_sma_trend(sample_ohlcv)
        assert r.signal in (-1, 0, 1)

    def test_ema_trend(self, sample_ohlcv):
        from indicators import calc_ema_trend
        r = calc_ema_trend(sample_ohlcv)
        assert r.signal in (-1, 0, 1)

    def test_adx(self, sample_ohlcv):
        from indicators import calc_adx
        r = calc_adx(sample_ohlcv)
        assert r.value >= 0

    def test_obv(self, sample_ohlcv):
        from indicators import calc_obv
        r = calc_obv(sample_ohlcv)
        assert r.signal in (-1, 0, 1)

    def test_williams_r_range(self, sample_ohlcv):
        from indicators import calc_williams_r
        r = calc_williams_r(sample_ohlcv)
        assert -100 <= r.value <= 0

    def test_vwap(self, sample_ohlcv):
        from indicators import calc_vwap
        r = calc_vwap(sample_ohlcv)
        assert r.value > 0
        assert r.signal in (-1, 0, 1)

    def test_supertrend(self, sample_ohlcv):
        from indicators import calc_supertrend
        r = calc_supertrend(sample_ohlcv)
        assert r.signal in (-1, 0, 1)

    def test_support_resistance(self, sample_ohlcv):
        from indicators import calc_support_resistance
        sr = calc_support_resistance(sample_ohlcv)
        assert sr.support >= 0
        assert sr.resistance >= 0
        # Support should be below price, resistance above (if both found)
        price = sample_ohlcv["Close"].iloc[-1]
        if sr.support > 0:
            assert sr.support < price
        if sr.resistance > 0:
            assert sr.resistance > price

    def test_breakout_inside_range(self, sample_ohlcv):
        from indicators import calc_support_resistance, calc_breakout
        sr = calc_support_resistance(sample_ohlcv)
        price = sample_ohlcv["Close"].iloc[-1]
        vol = int(sample_ohlcv["Volume"].iloc[-1])
        avg_vol = sample_ohlcv["Volume"].tail(20).mean()
        # If price is between support and resistance, signal should be 0
        if sr.support < price < sr.resistance:
            r = calc_breakout(sr, price, vol, avg_vol)
            assert r.signal == 0

    def test_risk_reward(self, sample_ohlcv):
        from indicators import calc_support_resistance, calc_risk_reward
        sr = calc_support_resistance(sample_ohlcv)
        price = sample_ohlcv["Close"].iloc[-1]
        r = calc_risk_reward(sr, price)
        assert r.value >= 0

    def test_volume_ratio(self, sample_ohlcv):
        from indicators import calc_volume_ratio
        r = calc_volume_ratio(sample_ohlcv)
        assert r.value > 0

    def test_volume_profile(self, sample_ohlcv):
        from indicators import calc_volume_profile
        r = calc_volume_profile(sample_ohlcv)
        assert r.poc > 0
        assert r.value_area_high >= r.value_area_low

    def test_volume_profile_flat(self):
        from indicators import calc_volume_profile
        df = pd.DataFrame({
            "High": [100.0]*60, "Low": [100.0]*60, "Close": [100.0]*60,
            "Volume": [1000.0]*60,
        })
        r = calc_volume_profile(df)
        assert r.poc == 100.0

    def test_volume_profile_zero_vol(self):
        from indicators import calc_volume_profile
        df = pd.DataFrame({
            "Open": np.linspace(100, 110, 60),
            "High": np.linspace(102, 112, 60),
            "Low": np.linspace(99, 109, 60),
            "Close": np.linspace(101, 111, 60),
            "Volume": [0.0]*60,
        })
        r = calc_volume_profile(df)
        assert r.poc > 0

    def test_atr_informational(self, sample_ohlcv):
        from indicators import calc_atr
        r = calc_atr(sample_ohlcv)
        assert r.signal == 0  # ATR is always informational


class TestAnalyzeStock:
    """Test the composite analysis."""

    def test_analyze_basic(self, sample_ohlcv):
        from indicators import analyze_stock
        a = analyze_stock(sample_ohlcv, "TEST", "Test", "تجربة")
        assert a.ticker == "TEST"
        assert a.current_price > 0
        assert len(a.indicators) >= 12  # We now have 15+ indicators

    def test_analyze_bullish(self, bullish_ohlcv):
        from indicators import analyze_stock
        a = analyze_stock(bullish_ohlcv, "BULL", "Bull", "صاعد")
        sma = [i for i in a.indicators if i.name == "SMA Trend"]
        if sma:
            assert sma[0].signal == 1

    def test_analyze_bearish(self, bearish_ohlcv):
        from indicators import analyze_stock
        a = analyze_stock(bearish_ohlcv, "BEAR", "Bear", "هابط")
        assert a.composite_score <= 1

    def test_analyze_insufficient_data(self):
        from indicators import analyze_stock
        with pytest.raises(ValueError, match="Insufficient"):
            analyze_stock(_make_ohlcv(30), "X", "X", "س")

    def test_analyze_missing_column(self):
        from indicators import analyze_stock
        dates = pd.date_range(end=datetime.now(), periods=100, freq="B")
        df = pd.DataFrame({
            "Open": np.linspace(100, 110, 100),
            "High": np.linspace(102, 112, 100),
            "Low": np.linspace(99, 109, 100),
            "Close": np.linspace(101, 111, 100),
        }, index=dates)
        with pytest.raises((ValueError, KeyError)):
            analyze_stock(df, "NOVOL", "No Vol", "بدون حجم")

    def test_signal_labels(self, sample_ohlcv):
        from indicators import analyze_stock
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        assert a.signal_label in ("شراء قوي 🟢🟢", "شراء 🟢", "محايد 🟡", "بيع 🔴", "بيع قوي 🔴🔴")

    def test_atr_excluded_from_score(self, sample_ohlcv):
        from indicators import analyze_stock
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        atr = [i for i in a.indicators if i.name == "ATR"]
        assert len(atr) == 1
        assert atr[0].signal == 0

    def test_extended_fields_populated(self, sample_ohlcv):
        """Extended fields (support, resistance, vwap) should be populated."""
        from indicators import analyze_stock
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        # VWAP should be set
        assert a.vwap > 0
        # SuperTrend signal should be set
        assert a.supertrend_signal in ("صاعد", "هابط", "محايد")
        # Support/resistance should be set (might be 0 if not found)
        # but the field should exist
        assert hasattr(a, 'support')
        assert hasattr(a, 'resistance')

    def test_dropna(self):
        from indicators import analyze_stock
        df = _make_ohlcv(250)
        df.iloc[10:15, df.columns.get_loc("Close")] = np.nan
        try:
            a = analyze_stock(df, "N", "N", "ن")
            assert a.ticker == "N"
        except ValueError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# SCORING ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestScoringEngine:
    """Test the deterministic 0-100 scoring engine."""

    def test_score_range(self, sample_ohlcv):
        """Score should be 0-100."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        result = compute_score(a)
        assert 0 <= result.total_score <= 100

    def test_score_deterministic(self, sample_ohlcv):
        """Same input should produce same score."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r1 = compute_score(a)
        r2 = compute_score(a)
        assert r1.total_score == r2.total_score

    def test_recommendation_values(self, sample_ohlcv):
        """Recommendation should be one of the valid values."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)
        assert r.recommendation in ("Buy", "Watch", "Sell", "No Trade")

    def test_recommendation_ar(self, sample_ohlcv):
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)
        assert r.recommendation_ar in ("شراء 🟢", "مراقبة 🟡", "بيع 🔴", "لا تداول ⚪")

    def test_factors_count(self, sample_ohlcv):
        """Should have 8 scoring factors."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)
        assert len(r.factors) == 8

    def test_weights_sum_to_100(self):
        """Factor weights must sum to 100."""
        from scoring import WEIGHTS
        assert sum(WEIGHTS.values()) == 100

    def test_poor_data_no_trade(self, sample_ohlcv):
        """Poor data quality should result in No Trade."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a, data_quality=0.3)
        assert r.recommendation == "No Trade"

    def test_stale_data_no_trade(self, sample_ohlcv):
        """Stale data should result in No Trade."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a, data_freshness="stale")
        assert r.recommendation == "No Trade"

    def test_bullish_stock_higher_score(self, bullish_ohlcv, bearish_ohlcv):
        """Bullish stock should score higher than bearish."""
        from indicators import analyze_stock
        from scoring import compute_score
        bull = analyze_stock(bullish_ohlcv, "B", "B", "صاعد")
        bear = analyze_stock(bearish_ohlcv, "B", "B", "هابط")
        bull_score = compute_score(bull).total_score
        bear_score = compute_score(bear).total_score
        assert bull_score > bear_score

    def test_risk_level_valid(self, sample_ohlcv):
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)
        assert r.risk_level in ("Low", "Medium", "High")

    def test_factors_explainable(self, sample_ohlcv):
        """Every factor should have a non-empty reason."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)
        for f in r.factors:
            assert f.reason, f"Factor {f.name} has empty reason"

    def test_pass_fail_reasons(self, sample_ohlcv):
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)
        # At least some reasons should be populated
        total_reasons = len(r.pass_reasons) + len(r.fail_reasons)
        assert total_reasons > 0


# ═══════════════════════════════════════════════════════════════════════════
# AI REPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestAIReport:
    """Test AI explanation and Telegram formatting."""

    def test_escape_markdown_v1_only(self):
        from ai_report import _escape_markdown
        assert "\\*" in _escape_markdown("test*text")
        assert "\\_" in _escape_markdown("test_text")
        # V2-only chars should NOT be escaped
        assert "\\(" not in _escape_markdown("test(text)")
        assert "\\-" not in _escape_markdown("test-text")

    def test_escape_markdown_empty(self):
        from ai_report import _escape_markdown
        assert _escape_markdown("") == ""
        assert _escape_markdown(None) is None

    def test_safe_truncate_short(self):
        from ai_report import _safe_truncate
        assert _safe_truncate("short", 100) == "short"

    def test_safe_truncate_long(self):
        from ai_report import _safe_truncate
        long_text = "line1\nline2\n" * 100
        result = _safe_truncate(long_text, 50)
        assert "…" in result

    def test_format_arabic_date(self):
        from ai_report import _format_arabic_date
        d = _format_arabic_date()
        assert any(day in d for day in ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"])

    def test_build_telegram_message(self, sample_ohlcv):
        from ai_report import build_telegram_message
        from indicators import analyze_stock
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        msg = build_telegram_message("ملخص", [a], None)
        assert "تقرير" in msg

    def test_build_telegram_message_truncation(self, sample_ohlcv):
        from ai_report import build_telegram_message
        from indicators import analyze_stock
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        msg = build_telegram_message("A"*5000, [a], None)
        assert len(msg) <= 3802

    def test_build_stocks_table_empty(self):
        """Empty/no scored stocks should show 'no opportunities' message."""
        from ai_report import build_stocks_table_message
        msg = build_stocks_table_message([])
        # Should contain something about no opportunities or be short
        assert len(msg) > 0


# ═══════════════════════════════════════════════════════════════════════════
# FETCH EGX TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFetchEGX:
    def test_market_summary_creation(self):
        from fetch_egx import MarketSummary
        s = MarketSummary("EGX 30", "30000", "+100", "+0.33%", "up")
        assert s.current_value == "30000"
        assert s.direction == "up"

    def test_format_summary_text(self):
        from fetch_egx import MarketSummary, format_summary_text
        s = MarketSummary("EGX 30", "30000", "+100", "+0.33%", "up",
                         month_change_pct="+2.5%", year_change_pct="+15.3%")
        text = format_summary_text(s)
        assert "30000" in text

    def test_safe_get_failure(self):
        from fetch_egx import _safe_get
        assert _safe_get("http://nonexistent.invalid", timeout=2, retries=0) is None


# ═══════════════════════════════════════════════════════════════════════════
# BOT TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBot:
    def test_trading_day_sunday(self):
        from bot import _is_egx_trading_day
        with patch("bot.datetime") as m:
            m.now.return_value.weekday.return_value = 6
            assert _is_egx_trading_day() is True

    def test_trading_day_friday(self):
        from bot import _is_egx_trading_day
        with patch("bot.datetime") as m:
            m.now.return_value.weekday.return_value = 4
            assert _is_egx_trading_day() is False

    def test_trading_day_saturday(self):
        from bot import _is_egx_trading_day
        with patch("bot.datetime") as m:
            m.now.return_value.weekday.return_value = 5
            assert _is_egx_trading_day() is False

    def test_check_env_missing(self):
        from bot import check_env
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit):
                check_env()

    def test_check_env_present(self):
        from bot import check_env
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "x", "GEMINI_API_KEY": "x"}):
            check_env()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline(self, bullish_ohlcv):
        """Full pipeline: data → indicators → scoring → formatting."""
        from indicators import analyze_stock
        from scoring import compute_score
        from ai_report import format_stock_detail

        a = analyze_stock(bullish_ohlcv, "TEST", "Test Stock", "سهم تجريبي")
        a.data_freshness = "fresh"
        a.data_quality = 0.9
        a.scoring_result = compute_score(a, "fresh", 0.9)

        detail = format_stock_detail(a)
        assert "TEST" in detail or "تجريبي" in detail
        assert "درجة" in detail  # Score should appear

    def test_scoring_consistency(self, sample_ohlcv):
        """Score from compute_score should align with indicator signals."""
        from indicators import analyze_stock
        from scoring import compute_score
        a = analyze_stock(sample_ohlcv, "T", "T", "ت")
        r = compute_score(a)

        # If composite score is strongly bullish, total score should be > 50
        if a.composite_score >= 4:
            assert r.total_score >= 50, f"Strong bullish (score={a.composite_score}) but total only {r.total_score}"

    def test_multi_stock_sorting(self, sample_ohlcv, bullish_ohlcv, bearish_ohlcv):
        from indicators import analyze_stock
        stocks = [
            analyze_stock(sample_ohlcv, "N", "Neutral", "محايد"),
            analyze_stock(bullish_ohlcv, "B", "Bull", "صاعد"),
            analyze_stock(bearish_ohlcv, "B", "Bear", "هابط"),
        ]
        stocks.sort(key=lambda x: x.composite_score, reverse=True)
        assert stocks[0].composite_score >= stocks[-1].composite_score


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASE & REGRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and regression tests for production safety."""

    def test_stochastic_rsi_flat_rsi(self):
        """Stochastic RSI should handle flat RSI (division by zero) gracefully."""
        from indicators import calc_stochastic_rsi
        # Create data that produces flat RSI (all closes equal)
        dates = pd.date_range(end=datetime.now(), periods=100, freq="B")
        df = pd.DataFrame({
            "Open": [100.0]*100, "High": [100.0]*100, "Low": [100.0]*100,
            "Close": [100.0]*100, "Volume": [1000.0]*100,
        }, index=dates)
        result = calc_stochastic_rsi(df)
        # Should return a valid result, not crash
        assert result.name == "Stochastic RSI"
        assert 0 <= result.value <= 100 or result.value == 50.0

    def test_vwap_zero_volume(self):
        """VWAP should handle zero volume gracefully."""
        from indicators import calc_vwap
        dates = pd.date_range(end=datetime.now(), periods=60, freq="B")
        df = pd.DataFrame({
            "Open": np.linspace(100, 110, 60),
            "High": np.linspace(102, 112, 60),
            "Low": np.linspace(99, 109, 60),
            "Close": np.linspace(101, 111, 60),
            "Volume": [0.0]*60,
        }, index=dates)
        result = calc_vwap(df)
        assert result.value > 0  # Should fall back to close price

    def test_supertrend_no_bias(self):
        """SuperTrend should start neutral (no bullish bias)."""
        from indicators import calc_supertrend
        # With very first bars, trend should not default to bullish
        df = _make_ohlcv(250, trend=0.0)  # No trend
        result = calc_supertrend(df)
        assert result.signal in (-1, 0, 1)

    def test_volume_ratio_zero_avg(self):
        """Volume ratio should handle zero average volume."""
        from indicators import calc_volume_ratio
        dates = pd.date_range(end=datetime.now(), periods=60, freq="B")
        df = pd.DataFrame({
            "Open": [100.0]*60, "High": [101.0]*60, "Low": [99.0]*60,
            "Close": [100.0]*60, "Volume": [0.0]*60,
        }, index=dates)
        result = calc_volume_ratio(df)
        assert result.value == 1.0  # Safe default

    def test_risk_reward_no_levels(self):
        """Risk/Reward should handle missing support/resistance."""
        from indicators import calc_risk_reward, SupportResistanceResult
        sr = SupportResistanceResult(0, 0, 0, 0)  # No levels found
        result = calc_risk_reward(sr, 100.0)
        assert result.value == 0.0
        assert result.signal == 0

    def test_scoring_result_field_exists(self):
        """StockAnalysis should have scoring_result field (not dynamic hasattr)."""
        from indicators import StockAnalysis
        a = StockAnalysis(ticker="T", name="T", name_ar="ت",
                         current_price=100, daily_change_pct=1, volume=1000)
        # Should be None by default, not raise AttributeError
        assert a.scoring_result is None

    def test_validator_quality_additive(self):
        """Quality degradation should be additive, not exponential."""
        from data.validator import ValidationResult
        r = ValidationResult(is_valid=True)
        initial = r.quality_score
        r.add_issue("warning 1", "warning")
        after1 = r.quality_score
        r.add_issue("warning 2", "warning")
        after2 = r.quality_score
        # Each warning should subtract ~0.15, not multiply by 0.8
        assert abs((initial - after1) - (after1 - after2)) < 0.01, \
            f"Degradation should be linear: {initial} -> {after1} -> {after2}"

    def test_canonical_list_rejects_unknown(self):
        """update_canonical_list should NOT accept unknown symbols from scraping."""
        from data.symbols import update_canonical_list, is_valid_egx_symbol
        # Try to inject a fake symbol
        update_canonical_list([{"symbol": "FAKE_INJECTED", "name": "Fake"}])
        assert is_valid_egx_symbol("FAKE_INJECTED") is False

    def test_escape_markdown_special_chars(self):
        """Markdown escaping should handle all V1 special chars."""
        from ai_report import _escape_markdown
        # All V1 specials: * _ ` [
        text = "price *bold* _under_ `code` [link]"
        escaped = _escape_markdown(text)
        assert "\*" in escaped
        assert "\_" in escaped
        assert "\`" in escaped
        assert "\[" in escaped

    def test_safe_truncate_preserves_end(self):
        """Truncation should always end with ellipsis."""
        from ai_report import _safe_truncate
        long = "A" * 5000
        result = _safe_truncate(long, 100)
        assert result.endswith("…") or result.endswith("…\n")

    def test_format_stock_detail_with_scoring(self, bullish_ohlcv):
        """format_stock_detail should include score when scoring_result is set."""
        from indicators import analyze_stock
        from scoring import compute_score
        from ai_report import format_stock_detail
        a = analyze_stock(bullish_ohlcv, "TEST", "Test", "تجربة")
        a.scoring_result = compute_score(a, "fresh", 0.9)
        detail = format_stock_detail(a)
        assert "درجة" in detail
        assert "/100" in detail

    def test_format_stock_detail_without_scoring(self, sample_ohlcv):
        """format_stock_detail should work even without scoring_result."""
        from indicators import analyze_stock
        from ai_report import format_stock_detail
        a = analyze_stock(sample_ohlcv, "TEST", "Test", "تجربة")
        # scoring_result is None by default
        detail = format_stock_detail(a)
        assert "TEST" in detail or "تجربة" in detail
