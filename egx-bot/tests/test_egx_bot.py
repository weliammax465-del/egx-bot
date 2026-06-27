"""
tests/test_egx_bot.py
---------------------
Unit tests for the EGX Daily Market Bot.
Tests run without real API keys or network access (all mocked).

Run: python -m pytest tests/ -v
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_egx import (
    MarketSummary,
    build_market_summary,
    format_summary_text,
    _parse_number,
    _safe_get,
)
from ai_report import (
    generate_arabic_report,
    build_telegram_message,
    _escape_markdown,
    _format_arabic_date,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_summary():
    """A realistic MarketSummary for testing."""
    return MarketSummary(
        index_name="EGX 30",
        current_value="51,443.07",
        change="-267.83",
        change_pct="-0.52%",
        direction="down",
        month_change_pct="-2.67%",
        year_change_pct="54.92%",
        date_str="Jun/25",
        source_note="البيانات من Trading Economics – للأغراض المعلوماتية فقط.",
        is_trading_day=True,
    )


@pytest.fixture
def empty_summary():
    """An empty MarketSummary (market closed / source failure)."""
    return MarketSummary(
        index_name="EGX 30",
        current_value="N/A",
        change="N/A",
        change_pct="N/A",
        direction="flat",
        source_note="تعذّر جلب البيانات.",
        is_trading_day=False,
    )


# ─── fetch_egx.py Tests ──────────────────────────────────────────────────────

class TestParseNumber:
    def test_positive_with_commas(self):
        assert _parse_number("51,443.07") == pytest.approx(51443.07)

    def test_negative(self):
        assert _parse_number("-267.83") == pytest.approx(-267.83)

    def test_empty(self):
        assert _parse_number("") is None

    def test_garbage(self):
        assert _parse_number("N/A") is None

    def test_plain_number(self):
        assert _parse_number("42") == pytest.approx(42.0)

    def test_none(self):
        assert _parse_number(None) is None


class TestSafeGet:
    @patch("fetch_egx.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        result = _safe_get("http://example.com", retries=0)
        assert result is mock_resp

    @patch("fetch_egx.requests.get")
    @patch("fetch_egx.time.sleep")
    def test_retry_then_success(self, mock_sleep, mock_get):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = Exception("503")
        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [fail_resp, ok_resp]
        result = _safe_get("http://example.com", retries=1)
        assert result is ok_resp
        assert mock_get.call_count == 2

    @patch("fetch_egx.requests.get")
    @patch("fetch_egx.time.sleep")
    def test_all_retries_fail(self, mock_sleep, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        result = _safe_get("http://example.com", retries=2)
        assert result is None
        assert mock_get.call_count == 3  # initial + 2 retries


class TestFormatSummaryText:
    def test_full_summary(self, mock_summary):
        text = format_summary_text(mock_summary)
        assert "EGX 30 Index: 51,443.07 📉" in text
        assert "Change: -267.83" in text
        assert "Monthly Change: -2.67%" in text
        assert "Yearly Change: 54.92%" in text

    def test_empty_summary(self, empty_summary):
        text = format_summary_text(empty_summary)
        assert "EGX 30 Index: N/A" in text
        # No monthly/yearly for empty
        assert "Monthly" not in text


class TestBuildMarketSummary:
    @patch("fetch_egx.fetch_egx30_index")
    def test_with_data(self, mock_index):
        mock_index.return_value = {
            "value": "51,443.07",
            "change": "-267.83",
            "change_pct": "-0.52%",
            "month_change_pct": "-2.67%",
            "year_change_pct": "54.92%",
            "direction": "down",
            "date_str": "Jun/25",
        }
        summary = build_market_summary()
        assert summary.current_value == "51,443.07"
        assert summary.direction == "down"
        assert summary.month_change_pct == "-2.67%"
        assert summary.is_trading_day is True

    @patch("fetch_egx.fetch_egx30_index")
    def test_no_data(self, mock_index):
        mock_index.return_value = {}
        summary = build_market_summary()
        assert summary.current_value == "N/A"
        assert summary.is_trading_day is False


class TestFetchEgx30Index:
    @patch("fetch_egx._safe_get")
    def test_parses_trading_economics_table(self, mock_get):
        """Test that we can parse the Trading Economics HTML table format."""
        from bs4 import BeautifulSoup
        mock_resp = MagicMock()
        # Simulate Trading Economics HTML table
        html = """
        <table>
            <tr><th>Indexes</th><th>Price</th><th></th><th></th><th>Day</th><th>Month</th><th>Year</th><th>Date</th></tr>
            <tr>
                <td>EGX 30</td><td>51,443.07</td><td></td>
                <td>-267.83</td><td>-0.52%</td><td>-2.67%</td><td>54.92%</td>
                <td>Jun/25</td>
            </tr>
        </table>
        """
        mock_resp.text = html
        mock_get.return_value = mock_resp

        from fetch_egx import fetch_egx30_index
        data = fetch_egx30_index()
        assert data["value"] == "51,443.07"
        assert data["change"] == "-267.83"
        assert data["change_pct"] == "-0.52%"
        assert data["direction"] == "down"

    @patch("fetch_egx._safe_get")
    def test_returns_empty_on_failure(self, mock_get):
        mock_get.return_value = None
        from fetch_egx import fetch_egx30_index
        data = fetch_egx30_index()
        assert data == {}


# ─── ai_report.py Tests ──────────────────────────────────────────────────────

class TestEscapeMarkdown:
    def test_plain_text(self):
        assert _escape_markdown("Hello World") == "Hello World"

    def test_asterisk(self):
        assert _escape_markdown("CIB*") == "CIB\\*"

    def test_underscore(self):
        assert _escape_markdown("EGX_30") == "EGX\\_30"

    def test_brackets(self):
        assert _escape_markdown("test[0]") == "test\\[0\\]"

    def test_empty(self):
        assert _escape_markdown("") == ""

    def test_none(self):
        assert _escape_markdown(None) is None


class TestFormatArabicDate:
    def test_returns_arabic(self):
        date_str = _format_arabic_date()
        arabic_months = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                         "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
        assert any(month in date_str for month in arabic_months)

    def test_contains_arabic_day(self):
        date_str = _format_arabic_date()
        arabic_days = ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس",
                       "الجمعة", "السبت", "الأحد"]
        assert any(day in date_str for day in arabic_days)

    def test_contains_year(self):
        date_str = _format_arabic_date()
        assert "2026" in date_str


class TestGenerateArabicReport:
    def test_missing_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError):
                generate_arabic_report("test data")

    @patch("ai_report.genai")
    def test_successful_generation(self, mock_genai):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            mock_model = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "هذا ملخص عربي للسوق اليوم."
            mock_model.generate_content.return_value = mock_response
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model

            result = generate_arabic_report("EGX 30: 51443 -267")
            assert "ملخص عربي" in result
            mock_model.generate_content.assert_called_once()

    @patch("ai_report.genai")
    @patch("ai_report.time.sleep")
    def test_fallback_on_failure(self, mock_sleep, mock_genai):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("API Error")
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model

            result = generate_arabic_report("EGX 30: 51443")
            assert "تعذّر" in result or "EGX 30" in result


class TestBuildTelegramMessage:
    def test_full_message(self, mock_summary):
        msg = build_telegram_message("ملخص عربي هنا", mock_summary)
        assert "تقرير البورصة المصرية اليومي" in msg
        assert "EGX 30" in msg
        assert "ملخص الذكاء الاصطناعي" in msg
        assert "نصيحة استثمارية" in msg

    def test_empty_market(self, empty_summary):
        msg = build_telegram_message("لا توجد بيانات", empty_summary)
        assert "N/A" in msg
        assert "تقرير البورصة المصرية اليومي" in msg

    def test_message_under_4096(self, mock_summary):
        long_report = "A" * 5000
        msg = build_telegram_message(long_report, mock_summary)
        assert len(msg) <= 4096

    def test_markdown_escaping(self):
        """Stock names with special chars should be escaped."""
        summary = MarketSummary(
            index_name="EGX 30",
            current_value="100",
            change="+1",
            change_pct="1%",
            direction="up",
            top_gainers=[{"name": "TEST_STOCK", "price": "10", "change_pct": "+5%"}],
        )
        msg = build_telegram_message("summary", summary)
        assert "TEST\\_STOCK" in msg


# ─── bot.py Tests ────────────────────────────────────────────────────────────

class TestTradingDayCheck:
    def test_friday_is_not_trading(self):
        from bot import _is_egx_trading_day
        friday = datetime(2026, 6, 26, 10, 0, 0)  # Friday
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = friday
            assert _is_egx_trading_day() is False

    def test_saturday_is_not_trading(self):
        from bot import _is_egx_trading_day
        saturday = datetime(2026, 6, 27, 10, 0, 0)  # Saturday
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            assert _is_egx_trading_day() is False

    def test_sunday_is_trading(self):
        from bot import _is_egx_trading_day
        sunday = datetime(2026, 6, 28, 10, 0, 0)  # Sunday
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = sunday
            assert _is_egx_trading_day() is True

    def test_thursday_is_trading(self):
        from bot import _is_egx_trading_day
        thursday = datetime(2026, 7, 2, 10, 0, 0)  # Thursday
        with patch("bot.datetime") as mock_dt:
            mock_dt.now.return_value = thursday
            assert _is_egx_trading_day() is True


class TestCheckEnv:
    def test_missing_env_exits(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                from bot import check_env
                check_env()

    def test_present_env_passes(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "x", "GEMINI_API_KEY": "y"}):
            from bot import check_env
            check_env()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
