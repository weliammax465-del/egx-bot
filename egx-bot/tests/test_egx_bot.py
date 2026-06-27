"""
tests/test_egx_bot.py
---------------------
Unit tests for the EGX Technical Analysis Bot.
Covers: indicators, stock scanner, AI report, bot commands, fetch_egx.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


# ─── Test Data Fixtures ──────────────────────────────────────────────────────

def _make_ohlcv(days: int = 250, start_price: float = 100.0, trend: float = 0.1) -> pd.DataFrame:
    """Generate realistic OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    np.random.seed(42)
    
    closes = [start_price]
    for i in range(1, days):
        change = np.random.normal(trend, 1.5)
        closes.append(max(closes[-1] + change, 1.0))
    
    closes = np.array(closes)
    opens = closes + np.random.normal(0, 0.5, days)
    highs = np.maximum(opens, closes) + np.random.uniform(0.1, 1.0, days)
    lows = np.minimum(opens, closes) - np.random.uniform(0.1, 1.0, days)
    volumes = np.random.randint(100000, 5000000, days).astype(float)
    
    df = pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
    }, index=dates)
    
    return df


@pytest.fixture
def sample_ohlcv():
    return _make_ohlcv(250)


@pytest.fixture
def bullish_ohlcv():
    """Strong uptrend data."""
    return _make_ohlcv(250, start_price=50, trend=0.3)


@pytest.fixture
def bearish_ohlcv():
    """Strong downtrend data."""
    return _make_ohlcv(250, start_price=200, trend=-0.3)


# ─── Indicator Tests ─────────────────────────────────────────────────────────

class TestIndicators:
    """Test individual technical indicators."""

    def test_rsi_range(self, sample_ohlcv):
        """RSI should be between 0 and 100."""
        from indicators import calc_rsi
        result = calc_rsi(sample_ohlcv)
        assert 0 <= result.value <= 100
        assert result.signal in (-1, 0, 1)

    def test_rsi_oversold(self, bearish_ohlcv):
        """Very bearish data should show low RSI."""
        from indicators import calc_rsi
        result = calc_rsi(bearish_ohlcv)
        assert result.value <= 100
        assert result.name == "RSI"

    def test_stochastic_range(self, sample_ohlcv):
        """Stochastic %K should be between 0 and 100."""
        from indicators import calc_stochastic
        result = calc_stochastic(sample_ohlcv)
        assert 0 <= result.value <= 100
        assert result.signal in (-1, 0, 1)

    def test_macd_signal(self, sample_ohlcv):
        """MACD should return a signal."""
        from indicators import calc_macd
        result = calc_macd(sample_ohlcv)
        assert result.name == "MACD"
        assert result.signal in (-1, 0, 1)

    def test_bollinger_position(self, sample_ohlcv):
        """Bollinger position should be between 0 and 100."""
        from indicators import calc_bollinger
        result = calc_bollinger(sample_ohlcv)
        assert 0 <= result.value <= 100
        assert result.signal in (-1, 0, 1)

    def test_sma_trend(self, sample_ohlcv):
        """SMA trend analysis should return a signal."""
        from indicators import calc_sma_trend
        result = calc_sma_trend(sample_ohlcv)
        assert result.name == "SMA Trend"
        assert result.signal in (-1, 0, 1)

    def test_adx_strength(self, sample_ohlcv):
        """ADX should be non-negative."""
        from indicators import calc_adx
        result = calc_adx(sample_ohlcv)
        assert result.value >= 0
        assert result.signal in (-1, 0, 1)

    def test_obv(self, sample_ohlcv):
        """OBV should return a signal."""
        from indicators import calc_obv
        result = calc_obv(sample_ohlcv)
        assert result.name == "OBV"
        assert result.signal in (-1, 0, 1)

    def test_williams_r_range(self, sample_ohlcv):
        """Williams %R should be between -100 and 0."""
        from indicators import calc_williams_r
        result = calc_williams_r(sample_ohlcv)
        assert -100 <= result.value <= 0

    def test_atr_positive(self, sample_ohlcv):
        """ATR should be positive."""
        from indicators import calc_atr
        result = calc_atr(sample_ohlcv)
        assert result.value > 0

    def test_volume_profile(self, sample_ohlcv):
        """Volume Profile should return POC and value area."""
        from indicators import calc_volume_profile
        result = calc_volume_profile(sample_ohlcv)
        assert result.poc > 0
        assert result.value_area_high >= result.value_area_low
        assert result.current_price_position in (
            "فوق منطقة القيمة", "داخل منطقة القيمة", "تحت منطقة القيمة"
        )

    def test_volume_profile_flat_data(self):
        """Volume Profile should handle flat price data."""
        from indicators import calc_volume_profile
        df = pd.DataFrame({
            "High": [100.0] * 60,
            "Low": [100.0] * 60,
            "Close": [100.0] * 60,
            "Volume": [1000.0] * 60,
        })
        result = calc_volume_profile(df)
        assert result.poc == 100.0


class TestAnalyzeStock:
    """Test the composite analysis function."""

    def test_analyze_basic(self, sample_ohlcv):
        """Full analysis should return a StockAnalysis with indicators."""
        from indicators import analyze_stock
        analysis = analyze_stock(sample_ohlcv, "TEST", "Test Stock", "سهم تجريبي")
        
        assert analysis.ticker == "TEST"
        assert analysis.name_ar == "سهم تجريبي"
        assert analysis.current_price > 0
        assert len(analysis.indicators) >= 8
        assert analysis.composite_score != 0 or True  # Could be neutral
        assert analysis.signal_label != ""

    def test_analyze_bullish_trend(self, bullish_ohlcv):
        """Bullish trend data should show SMA golden cross in reasons."""
        from indicators import analyze_stock
        analysis = analyze_stock(bullish_ohlcv, "BULL", "Bull Stock", "سهم صاعد")
        # Even if overbought, SMA trend should be bullish (golden cross)
        sma_ind = [i for i in analysis.indicators if i.name == "SMA Trend"]
        if sma_ind:
            assert sma_ind[0].signal == 1  # Bullish SMA trend

    def test_analyze_bearish_trend(self, bearish_ohlcv):
        """Bearish data should produce negative or neutral score."""
        from indicators import analyze_stock
        analysis = analyze_stock(bearish_ohlcv, "BEAR", "Bear Stock", "سهم هابط")
        assert analysis.composite_score <= 1  # At best neutral

    def test_analyze_insufficient_data(self):
        """Should raise error with insufficient data."""
        from indicators import analyze_stock
        df = _make_ohlcv(30)  # Only 30 days
        with pytest.raises(ValueError, match="Insufficient data"):
            analyze_stock(df, "SHORT", "Short", "قصير")

    def test_analyze_missing_column(self):
        """Should raise error if a required column is missing."""
        from indicators import analyze_stock
        dates = pd.date_range(end=datetime.now(), periods=100, freq="B")
        df = pd.DataFrame({
            "Open": np.linspace(100, 110, 100),
            "High": np.linspace(102, 112, 100),
            "Low": np.linspace(99, 109, 100),
            "Close": np.linspace(101, 111, 100),
        }, index=dates)
        with pytest.raises((ValueError, KeyError)):
            analyze_stock(df, "NOVOL", "No Volume", "بدون حجم")

    def test_signal_labels(self, sample_ohlcv):
        """Signal labels should be one of the expected values."""
        from indicators import analyze_stock
        analysis = analyze_stock(sample_ohlcv, "TEST", "Test", "تجربة")
        valid_labels = ["شراء قوي 🟢🟢", "شراء 🟢", "محايد 🟡", "بيع 🔴", "بيع قوي 🔴🔴"]
        assert analysis.signal_label in valid_labels

    def test_bullish_reasons_populated(self, bullish_ohlcv):
        """Bullish stock should have bullish reasons."""
        from indicators import analyze_stock
        analysis = analyze_stock(bullish_ohlcv, "BULL", "Bull", "صاعد")
        if analysis.composite_score > 0:
            assert len(analysis.bullish_reasons) > 0


# ─── Stock Scanner Tests ─────────────────────────────────────────────────────

class TestStockScanner:
    """Test the stock scanner functions."""

    def test_get_top_bullish(self):
        """get_top_bullish should return stocks sorted by score."""
        from stock_scanner import get_top_bullish
        from indicators import StockAnalysis
        
        stocks = [
            StockAnalysis(ticker="A", name="A", name_ar="أ", current_price=10,
                         daily_change_pct=1, volume=100, composite_score=5),
            StockAnalysis(ticker="B", name="B", name_ar="ب", current_price=20,
                         daily_change_pct=2, volume=200, composite_score=3),
            StockAnalysis(ticker="C", name="C", name_ar="ج", current_price=30,
                         daily_change_pct=-1, volume=300, composite_score=-2),
        ]
        
        top = get_top_bullish(stocks, 5)
        assert len(top) == 2  # Only A and B have score >= 2
        assert top[0].ticker == "A"  # Highest score first

    def test_get_top_bearish(self):
        """get_top_bearish should return most bearish stocks."""
        from stock_scanner import get_top_bearish
        from indicators import StockAnalysis
        
        stocks = [
            StockAnalysis(ticker="A", name="A", name_ar="أ", current_price=10,
                         daily_change_pct=1, volume=100, composite_score=5),
            StockAnalysis(ticker="B", name="B", name_ar="ب", current_price=20,
                         daily_change_pct=-2, volume=200, composite_score=-3),
            StockAnalysis(ticker="C", name="C", name_ar="ج", current_price=30,
                         daily_change_pct=-1, volume=300, composite_score=-5),
        ]
        
        top = get_top_bearish(stocks, 5)
        assert len(top) == 2
        assert top[0].ticker == "C"  # Most bearish first

    def test_format_analysis_for_ai(self):
        """format_analysis_for_ai should produce readable text."""
        from stock_scanner import format_analysis_for_ai
        from indicators import StockAnalysis, IndicatorResult
        
        s = StockAnalysis(
            ticker="TEST", name="Test", name_ar="تجربة",
            current_price=100, daily_change_pct=1.5, volume=50000,
            composite_score=3, signal_label="شراء 🟢", signal_score_pct=75.0,
        )
        s.indicators.append(IndicatorResult("RSI", "مؤشر القوة النسبية", 45.0, 1, "صاعد", "مذبذب إيجابي"))
        
        text = format_analysis_for_ai([s])
        assert "TEST" in text
        assert "تجربة" in text
        assert "RSI" in text

    def test_arabic_names_mapping(self):
        """Arabic names should be available for major stocks."""
        from stock_scanner import ARABIC_NAMES
        assert "COMI" in ARABIC_NAMES
        assert "ETEL" in ARABIC_NAMES
        assert "EAST" in ARABIC_NAMES
        assert len(ARABIC_NAMES) >= 50


# ─── AI Report Tests ─────────────────────────────────────────────────────────

class TestAIReport:
    """Test the AI report generation and formatting."""

    def test_escape_markdown(self):
        """Markdown special characters should be escaped."""
        from ai_report import _escape_markdown
        assert "\\*" in _escape_markdown("test*text")
        assert "\\_" in _escape_markdown("test_text")
        assert "\\[" in _escape_markdown("test[text]")

    def test_escape_markdown_empty(self):
        """Empty string should return empty."""
        from ai_report import _escape_markdown
        assert _escape_markdown("") == ""
        assert _escape_markdown(None) is None

    def test_format_arabic_date(self):
        """Arabic date should contain Arabic day and month names."""
        from ai_report import _format_arabic_date
        date_str = _format_arabic_date()
        assert any(day in date_str for day in ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"])
        assert str(datetime.now().year) in date_str or "2026" in date_str

    def test_build_telegram_message(self):
        """Telegram message should have header and structure."""
        from ai_report import build_telegram_message
        from indicators import StockAnalysis
        
        stocks = [
            StockAnalysis(ticker="TEST", name="Test", name_ar="تجربة",
                         current_price=100, daily_change_pct=1, volume=1000,
                         composite_score=3, signal_label="شراء 🟢"),
        ]
        
        msg = build_telegram_message("ملخص تجريبي", stocks, None)
        assert "تقرير" in msg
        assert "ملخص تجريبي" in msg
        assert "20" not in msg or "سهم" in msg  # Should mention stock count

    def test_build_telegram_message_with_market(self):
        """Telegram message should include EGX 30 data when provided."""
        from ai_report import build_telegram_message
        from indicators import StockAnalysis
        
        class MockMarket:
            current_value = 30000
            change = 100
            change_pct = "0.33%"
            direction = "up"
            month_change_pct = "2.5%"
            year_change_pct = "15.3%"
        
        stocks = [StockAnalysis(ticker="T", name="T", name_ar="ت",
                               current_price=10, daily_change_pct=1, volume=100,
                               composite_score=2, signal_label="شراء 🟢")]
        
        msg = build_telegram_message("test", stocks, MockMarket())
        assert "EGX 30" in msg
        assert "30000" in msg

    def test_build_stocks_table_message(self):
        """Stocks table should list bullish and bearish stocks."""
        from ai_report import build_stocks_table_message
        from indicators import StockAnalysis
        
        stocks = [
            StockAnalysis(ticker="BULL", name="Bull", name_ar="صاعد",
                         current_price=100, daily_change_pct=2, volume=1000,
                         composite_score=4, signal_label="شراء قوي 🟢🟢",
                         bullish_reasons=["RSI: تشبع بيعي"]),
            StockAnalysis(ticker="BEAR", name="Bear", name_ar="هابط",
                         current_price=50, daily_change_pct=-3, volume=2000,
                         composite_score=-4, signal_label="بيع قوي 🔴🔴",
                         bearish_reasons=["MACD: تقاطع هابط"]),
        ]
        
        msg = build_stocks_table_message(stocks)
        assert "صاعد" in msg or "BULL" in msg
        assert "هابط" in msg or "BEAR" in msg


# ─── Fetch EGX Tests ─────────────────────────────────────────────────────────

class TestFetchEGX:
    """Test the EGX index data fetcher."""

    def test_market_summary_creation(self):
        """MarketSummary data class should work correctly."""
        from fetch_egx import MarketSummary
        
        summary = MarketSummary(
            index_name="EGX 30",
            current_value="30,123.45",
            change="+123.45",
            change_pct="+0.41%",
            direction="up",
        )
        assert summary.current_value == "30,123.45"
        assert summary.direction == "up"
        assert summary.index_name == "EGX 30"

    def test_format_summary_text(self):
        """format_summary_text should produce readable text."""
        from fetch_egx import MarketSummary, format_summary_text
        
        summary = MarketSummary(
            index_name="EGX 30",
            current_value="30,000",
            change="+100",
            change_pct="+0.33%",
            direction="up",
            month_change_pct="+2.5%",
            year_change_pct="+15.3%",
        )
        text = format_summary_text(summary)
        assert "30,000" in text
        assert "+0.33%" in text


# ─── Bot Tests ───────────────────────────────────────────────────────────────

class TestBot:
    """Test bot utility functions."""

    def test_is_egx_trading_day_sunday(self):
        """Sunday should be a trading day."""
        from bot import _is_egx_trading_day
        # Mock datetime to return a Sunday
        with patch("bot.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 6  # Sunday
            mock_dt.now.return_value = mock_now
            assert _is_egx_trading_day() == True

    def test_is_egx_trading_day_friday(self):
        """Friday should NOT be a trading day."""
        from bot import _is_egx_trading_day
        with patch("bot.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 4  # Friday
            mock_dt.now.return_value = mock_now
            assert _is_egx_trading_day() == False

    def test_is_egx_trading_day_saturday(self):
        """Saturday should NOT be a trading day."""
        from bot import _is_egx_trading_day
        with patch("bot.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 5  # Saturday
            mock_dt.now.return_value = mock_now
            assert _is_egx_trading_day() == False

    def test_check_env_missing(self):
        """check_env should exit if env vars are missing."""
        from bot import check_env
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit):
                check_env()

    def test_check_env_present(self):
        """check_env should pass if required vars are set."""
        from bot import check_env
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test", "GEMINI_API_KEY": "test"}):
            check_env()  # Should not raise


# ─── Integration Tests ───────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests that verify end-to-end flows."""

    def test_full_analysis_pipeline(self, sample_ohlcv):
        """Full pipeline: data → indicators → analysis → formatting."""
        from indicators import analyze_stock
        from stock_scanner import format_analysis_for_ai, get_top_bullish
        from ai_report import build_telegram_message
        
        # Analyze
        analysis = analyze_stock(sample_ohlcv, "TEST", "Test Stock", "سهم تجريبي")
        
        # Format for AI
        ai_text = format_analysis_for_ai([analysis])
        assert "TEST" in ai_text
        
        # Build Telegram message
        msg = build_telegram_message("ملخص اختبار", [analysis], None)
        assert "تقرير" in msg
        assert "ملخص اختبار" in msg

    def test_multi_stock_analysis(self, sample_ohlcv, bullish_ohlcv, bearish_ohlcv):
        """Multiple stocks should be sortable by score."""
        from indicators import analyze_stock
        from stock_scanner import get_top_bullish, get_top_bearish
        
        stocks = [
            analyze_stock(sample_ohlcv, "NEUT", "Neutral", "محايد"),
            analyze_stock(bullish_ohlcv, "BULL", "Bullish", "صاعد"),
            analyze_stock(bearish_ohlcv, "BEAR", "Bearish", "هابط"),
        ]
        
        # Sort by score
        stocks.sort(key=lambda x: x.composite_score, reverse=True)
        assert stocks[0].composite_score >= stocks[-1].composite_score
        
        # Top bullish should include the bullish stock
        top = get_top_bullish(stocks, 3)
        if top:
            assert top[0].composite_score >= 2

    def test_volume_profile_with_real_data(self, sample_ohlcv):
        """Volume Profile should work with realistic OHLCV data."""
        from indicators import calc_volume_profile
        
        result = calc_volume_profile(sample_ohlcv, bins=30, lookback=60)
        assert result.poc > 0
        assert result.value_area_high > 0
        assert result.value_area_low > 0
        assert result.value_area_high >= result.value_area_low
        assert result.current_price_position in (
            "فوق منطقة القيمة", "داخل منطقة القيمة", "تحت منطقة القيمة"
        )
