"""
test_filters.py
---------------
Tests for the Liquidity-First pre-scoring gates (filters.py).

Uses synthetic but realistic EGX data to validate:
- Liquidity gate (turnover threshold)
- Price limit gate (circuit breaker)
- Combined gate (pass_all_gates)
- Edge cases (insufficient data, None, NaN)
"""

import pytest
import pandas as pd
import numpy as np

from filters import liquidity_gate, price_limit_gate, pass_all_gates, get_exclusion_reason, GateResult
import config


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_ohlcv(days: int = 60, base_price: float = 100.0, base_volume: int = 500_000) -> pd.DataFrame:
    """Generate a realistic OHLCV DataFrame."""
    dates = pd.bdate_range(start="2026-01-01", periods=days)
    # Small random walk for price
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_ = close * (1 + np.random.normal(0, 0.005, days))
    volume = base_volume + np.random.randint(-base_volume * 0.3, base_volume * 0.3, days)
    volume = np.maximum(volume, 1000)  # never zero

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)




def make_flat_ohlcv(days: int, price: float, volume: int) -> pd.DataFrame:
    """Generate a deterministic flat OHLCV DataFrame (no random walk)."""
    dates = pd.bdate_range(start="2026-01-01", periods=days)
    return pd.DataFrame({
        "Open": [price] * days, "High": [price * 1.01] * days,
        "Low": [price * 0.99] * days, "Close": [price] * days,
        "Volume": [volume] * days,
    }, index=dates)

# ─── Liquidity Gate Tests ────────────────────────────────────────────────────

class TestLiquidityGate:
    """Test the liquidity gate (MIN_TURNOVER_EGP = 1,000,000)."""

    def test_passes_with_high_turnover(self):
        """Stock with avg turnover > 1M EGP should pass."""
        # 50,000 shares × 50 EGP = 2.5M EGP/day → passes
        df = make_ohlcv(days=30, base_price=50.0, base_volume=50_000)
        result = liquidity_gate(df, "TEST")
        assert result.passed is True
        assert result.gate_name == "liquidity"
        assert "سيولة كافية" in result.reason

    def test_fails_with_low_turnover(self):
        """Stock with avg turnover < 1M EGP should fail."""
        # 1,000 shares × 5 EGP = 5,000 EGP/day → fails
        df = make_ohlcv(days=30, base_price=5.0, base_volume=1_000)
        result = liquidity_gate(df, "TEST")
        assert result.passed is False
        assert "سيولة ضعيفة" in result.reason
        assert result.details["avg_turnover_egp"] < config.MIN_TURNOVER_EGP

    def test_fails_with_insufficient_data(self):
        """Stock with < 20 days of data should fail."""
        df = make_ohlcv(days=10, base_price=100.0, base_volume=100_000)
        result = liquidity_gate(df, "TEST")
        assert result.passed is False
        assert "بيانات غير كافية" in result.reason

    def test_fails_with_none_dataframe(self):
        """None DataFrame should fail gracefully."""
        result = liquidity_gate(None, "TEST")
        assert result.passed is False
        assert result.reason_en == "insufficient data for liquidity avg (0 < 20 days)"

    def test_borderline_just_above_threshold(self):
        """Stock with turnover just above 1M should pass."""
        # 10,000 shares × 101 EGP = 1,010,000 EGP/day → passes
        df = make_flat_ohlcv(days=25, price=101.0, volume=10_000)
        result = liquidity_gate(df, "TEST")
        assert result.passed is True

    def test_borderline_just_below_threshold(self):
        """Stock with turnover just below 1M should fail."""
        # 9,800 shares × 101 EGP = 989,800 EGP/day → fails
        df = make_flat_ohlcv(days=25, price=101.0, volume=9_800)
        result = liquidity_gate(df, "TEST")
        assert result.passed is False

    def test_details_contain_turnover_values(self):
        """Gate result should include turnover details for logging."""
        df = make_ohlcv(days=25, base_price=50.0, base_volume=50_000)
        result = liquidity_gate(df, "TEST")
        assert "avg_turnover_egp" in result.details
        assert "latest_turnover_egp" in result.details
        assert "min_required_egp" in result.details
        assert result.details["min_required_egp"] == config.MIN_TURNOVER_EGP

    def test_custom_avg_days_override(self):
        """Should accept a custom avg_days parameter."""
        # With 5-day avg, need only 5 days of data
        df = make_ohlcv(days=10, base_price=50.0, base_volume=50_000)
        result = liquidity_gate(df, "TEST", avg_days=5)
        assert result.passed is True
        assert result.details["avg_days"] == 5


# ─── Price Limit Gate Tests ──────────────────────────────────────────────────

class TestPriceLimitGate:
    """Test the price limit gate (PRICE_LIMIT_THRESHOLD_PCT = 10.0)."""

    def test_passes_with_normal_change(self):
        """Stock with |change| < 10% should pass."""
        result = price_limit_gate(5.5, "TEST")
        assert result.passed is True
        assert "ضمن الحد الطبيعي" in result.reason

    def test_fails_at_positive_limit(self):
        """Stock at +10% should fail (circuit breaker hit)."""
        result = price_limit_gate(10.0, "TEST")
        assert result.passed is False
        assert "صعودي" in result.reason
        assert "مجمد" in result.reason

    def test_fails_at_negative_limit(self):
        """Stock at -10% should fail."""
        result = price_limit_gate(-10.0, "TEST")
        assert result.passed is False
        assert "هبوطي" in result.reason

    def test_fails_above_positive_limit(self):
        """Stock at +15% should fail."""
        result = price_limit_gate(15.0, "TEST")
        assert result.passed is False

    def test_passes_just_below_limit(self):
        """Stock at +9.9% should pass (still tradeable)."""
        result = price_limit_gate(9.9, "TEST")
        assert result.passed is True

    def test_passes_with_zero_change(self):
        """Stock with 0% change should pass."""
        result = price_limit_gate(0.0, "TEST")
        assert result.passed is True

    def test_fails_with_none_change(self):
        """None change_pct should fail."""
        result = price_limit_gate(None, "TEST")
        assert result.passed is False
        assert "غير متاحة" in result.reason

    def test_fails_with_nan_change(self):
        """NaN change_pct should fail."""
        result = price_limit_gate(float('nan'), "TEST")
        assert result.passed is False

    def test_negative_change_passes_below_limit(self):
        """Stock at -5% should pass."""
        result = price_limit_gate(-5.0, "TEST")
        assert result.passed is True


# ─── Combined Gate Tests ─────────────────────────────────────────────────────

class TestPassAllGates:
    """Test the combined gate function."""

    def test_passes_when_both_gates_pass(self):
        """Stock with good liquidity and normal change should pass all gates."""
        df = make_ohlcv(days=25, base_price=50.0, base_volume=50_000)
        passed, results = pass_all_gates(df, daily_change_pct=3.0, ticker="TEST")
        assert passed is True
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_fails_on_liquidity_only(self):
        """Stock with low liquidity should fail at the liquidity gate."""
        df = make_ohlcv(days=25, base_price=5.0, base_volume=1_000)
        passed, results = pass_all_gates(df, daily_change_pct=3.0, ticker="TEST")
        assert passed is False
        assert len(results) == 1  # fail fast — price limit not checked
        assert results[0].gate_name == "liquidity"

    def test_fails_on_price_limit_only(self):
        """Stock with good liquidity but at price limit should fail."""
        df = make_ohlcv(days=25, base_price=50.0, base_volume=50_000)
        passed, results = pass_all_gates(df, daily_change_pct=10.5, ticker="TEST")
        assert passed is False
        assert len(results) == 2  # both gates checked (liquidity passed first)
        assert results[0].passed is True  # liquidity
        assert results[1].passed is False  # price limit

    def test_fails_on_both_gates(self):
        """Stock failing both gates should stop at the first (liquidity)."""
        df = make_ohlcv(days=10, base_price=5.0, base_volume=1_000)
        passed, results = pass_all_gates(df, daily_change_pct=15.0, ticker="TEST")
        assert passed is False
        assert len(results) == 1  # fail fast

    def test_get_exclusion_reason_returns_first_failure(self):
        """get_exclusion_reason should return the first failing gate's reason."""
        df = make_ohlcv(days=10, base_price=5.0, base_volume=1_000)
        _, results = pass_all_gates(df, daily_change_pct=15.0, ticker="TEST")
        reason = get_exclusion_reason(results)
        assert "سيولة" in reason  # liquidity failed first

    def test_get_exclusion_reason_empty_when_all_pass(self):
        """get_exclusion_reason should return empty string when all pass."""
        df = make_ohlcv(days=25, base_price=50.0, base_volume=50_000)
        _, results = pass_all_gates(df, daily_change_pct=3.0, ticker="TEST")
        reason = get_exclusion_reason(results)
        assert reason == ""


# ─── Simulated EGX Scan (20 tracked stocks) ─────────────────────────────────

class TestSimulatedEGXScan:
    """
    Simulate the Liquidity Gate on the 20 tracked EGX stocks.

    Uses realistic price and volume estimates to show how many stocks
    would be excluded by the liquidity filter.
    """

    # Realistic EGX estimates: (ticker, price, avg_daily_volume, expected_to_pass)
    EGX_STOCKS = [
        # High liquidity — major stocks (pass)
        ("COMI",  150.0, 2_000_000, True),   # CIB — 300M EGP/day
        ("SWDY",   25.0, 8_000_000, True),   # El Sewedy — 200M EGP/day
        ("ORAS",  450.0,   300_000, True),   # Orascom — 135M EGP/day
        ("HRHO",    8.0, 15_000_000, True),  # Ezz Steel — 120M EGP/day
        ("ETEL",   30.0, 1_500_000, True),   # Telecom Egypt — 45M EGP/day
        ("ORWE",   20.0, 3_000_000, True),   # Orascom Wealth — 60M EGP/day
        ("CCAP",   15.0, 4_000_000, True),   # Credit Suisse — 60M EGP/day
        ("FWRY",   25.0, 2_000_000, True),   # Fawry — 50M EGP/day
        ("CIRA",   30.0, 1_500_000, True),   # CIRA — 45M EGP/day
        ("RAYA",   20.0, 3_000_000, True),   # Raya — 60M EGP/day
        ("AMOC",   30.0, 1_200_000, True),   # AMOC — 36M EGP/day
        ("ABUK",   40.0,   800_000, True),   # Abu Kir — 32M EGP/day
        ("MFPC",   15.0, 4_500_000, True),   # MFPC — 67.5M EGP/day
        # Low liquidity — small caps (fail)
        ("EKHO",    5.0,   150_000, False),  # Ekhlas — 750K EGP/day < 1M
        ("EFIH",    3.0,   200_000, False),  # EFIH — 600K EGP/day < 1M
        ("EFID",    2.5,   180_000, False),  # EFID — 450K EGP/day < 1M
        ("PHDC",    7.0,   120_000, False),  # Palm Hills — 840K EGP/day < 1M
        ("JUFO",   12.0,    70_000, False),  # Juhayna (low vol) — 840K EGP/day < 1M
        ("EAST",    3.0,   100_000, False),  # Eastern Co — 300K EGP/day < 1M
        ("SKPC",    2.0,   200_000, False),  # SKPC — 400K EGP/day < 1M
    ]

    def test_liquidity_gate_filters_correctly(self):
        """Run liquidity gate on all 20 stocks and verify pass/fail counts."""
        passed_count = 0
        failed_count = 0
        failed_tickers = []

        for ticker, price, avg_vol, expected_pass in self.EGX_STOCKS:
            df = make_ohlcv(days=25, base_price=price, base_volume=avg_vol)
            result = liquidity_gate(df, ticker)

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
                failed_tickers.append(ticker)

            # Verify our expectation matches
            assert result.passed == expected_pass, (
                f"{ticker}: expected {'pass' if expected_pass else 'fail'}, "
                f"got {'pass' if result.passed else 'fail'} "
                f"(avg turnover {result.details.get('avg_turnover_egp', 0)/1e6:.2f}M EGP)"
            )

        print(f"\n=== Simulated EGX Liquidity Gate Results ===")
        print(f"Total stocks: {len(self.EGX_STOCKS)}")
        print(f"Passed: {passed_count}")
        print(f"Excluded (low liquidity): {failed_count}")
        print(f"Excluded tickers: {', '.join(failed_tickers)}")
        print(f"============================================")

        assert passed_count == 13
        assert failed_count == 7
