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

from filters import (liquidity_gate, price_limit_gate, pass_all_gates, get_exclusion_reason, get_exclusion_code, GateResult, SurgeResult, ConfirmationResult, volume_surge_check, confirmation_check, trend_strength_filter, risk_filter)
import config
from scoring import compute_score_v2, WEIGHTS


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


# ─── Missing Volume Data Tests ───────────────────────────────────────────────

class TestMissingVolumeData:
    """
    Test that the liquidity gate distinguishes between:
    - "low liquidity" (have volume data, turnover is genuinely low)
    - "liquidity data unavailable" (Volume is zeros/NaN — can't assess)
    """

    def test_all_zero_volume_excluded_as_data_unavailable(self):
        """Stock with ALL Volume=0 should be 'data unavailable', NOT 'low liquidity'."""
        df = make_flat_ohlcv(days=25, price=50.0, volume=0)
        result = liquidity_gate(df, "TEST")
        assert result.passed is False
        assert "بيانات سيولة غير متوفرة" in result.reason
        assert result.exclusion_code == "بيانات سيولة غير متوفرة"
        # Must NOT say "low liquidity" — that's a different diagnosis
        assert "سيولة ضعيفة" not in result.reason

    def test_all_nan_volume_excluded_as_data_unavailable(self):
        """Stock with ALL Volume=NaN should be 'data unavailable'."""
        df = make_flat_ohlcv(days=25, price=50.0, volume=50_000)
        df["Volume"] = np.nan
        result = liquidity_gate(df, "TEST")
        assert result.passed is False
        assert "بيانات سيولة غير متوفرة" in result.reason
        assert result.exclusion_code == "بيانات سيولة غير متوفرة"

    def test_half_zero_volume_still_works(self):
        """If >=50% of days have valid volume, should compute from valid days."""
        df = make_flat_ohlcv(days=25, price=50.0, volume=50_000)
        # Zero out 10 of 25 days (40% zeroed → 60% valid → above 50% threshold)
        df.iloc[:10, df.columns.get_loc("Volume")] = 0
        result = liquidity_gate(df, "TEST")
        # Should pass because valid days have enough turnover
        assert result.passed is True
        assert result.details["valid_volume_days"] >= 10

    def test_mostly_zero_volume_excluded_as_data_unavailable(self):
        """If <50% of days have valid volume, should be 'data unavailable'."""
        df = make_flat_ohlcv(days=25, price=50.0, volume=50_000)
        # Zero out 15 of 25 days (60% zeroed → 40% valid → below 50% threshold)
        df.iloc[-15:, df.columns.get_loc("Volume")] = 0
        result = liquidity_gate(df, "TEST")
        assert result.passed is False
        assert "بيانات سيولة غير متوفرة" in result.reason

    def test_exclusion_codes_are_distinct(self):
        """Low liquidity and data unavailable must have different exclusion codes."""
        # Low liquidity
        df_low = make_flat_ohlcv(days=25, price=5.0, volume=1_000)
        result_low = liquidity_gate(df_low, "LOW")
        # Data unavailable
        df_zero = make_flat_ohlcv(days=25, price=50.0, volume=0)
        result_zero = liquidity_gate(df_zero, "ZERO")

        assert result_low.exclusion_code != result_zero.exclusion_code
        assert result_low.exclusion_code == "سيولة ضعيفة"
        assert result_zero.exclusion_code == "بيانات سيولة غير متوفرة"

    def test_get_exclusion_code_returns_correct_code(self):
        """get_exclusion_code should return the short code for analytics."""
        from filters import get_exclusion_code
        df = make_flat_ohlcv(days=25, price=50.0, volume=0)
        _, results = pass_all_gates(df, daily_change_pct=3.0, ticker="TEST")
        code = get_exclusion_code(results)
        assert code == "بيانات سيولة غير متوفرة"

    def test_get_exclusion_code_empty_when_all_pass(self):
        """get_exclusion_code should return empty string when all pass."""
        from filters import get_exclusion_code
        df = make_ohlcv(days=25, base_price=50.0, base_volume=50_000)
        _, results = pass_all_gates(df, daily_change_pct=3.0, ticker="TEST")
        code = get_exclusion_code(results)
        assert code == ""

    def test_low_liquidity_has_valid_volume_days_in_details(self):
        """Low liquidity result should report how many valid volume days were used."""
        df = make_flat_ohlcv(days=25, price=5.0, volume=1_000)
        result = liquidity_gate(df, "TEST")
        assert "valid_volume_days" in result.details
        assert result.details["valid_volume_days"] == 20  # all 20 days valid


# ─── Step 3: Volume Surge Tests ──────────────────────────────────────────────

class TestVolumeSurge:
    """Test the Volume Surge check (step 3)."""

    def test_surge_detected_when_today_highest(self):
        """Today's volume > all 3 prior days → surged=True."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=100_000)
        # Make today's volume higher
        df.iloc[-1, df.columns.get_loc("Volume")] = 200_000
        result = volume_surge_check(df, "TEST")
        assert result.surged is True
        assert result.days_surged == 3

    def test_no_surge_when_today_not_highest(self):
        """Today's volume not > all 3 prior days → surged=False."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=100_000)
        # Make yesterday's volume higher than today
        df.iloc[-2, df.columns.get_loc("Volume")] = 200_000
        result = volume_surge_check(df, "TEST")
        assert result.surged is False
        assert result.days_surged < 3

    def test_surge_partial_pass(self):
        """Today beats 2 of 3 days but not all → surged=False, days_surged=2."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=100_000)
        df.iloc[-1, df.columns.get_loc("Volume")] = 150_000  # today
        df.iloc[-2, df.columns.get_loc("Volume")] = 120_000  # day -1 (beaten)
        df.iloc[-3, df.columns.get_loc("Volume")] = 180_000  # day -2 (NOT beaten)
        df.iloc[-4, df.columns.get_loc("Volume")] = 130_000  # day -3 (beaten)
        result = volume_surge_check(df, "TEST")
        assert result.surged is False
        # Today beats day -1 and day -3 but not day -2
        assert result.days_surged == 2

    def test_surge_insufficient_data(self):
        """Less than 4 days of data → no surge."""
        df = make_flat_ohlcv(days=3, price=50.0, volume=100_000)
        result = volume_surge_check(df, "TEST")
        assert result.surged is False
        assert "غير كافية" in result.reason_ar

    def test_surge_none_dataframe(self):
        """None df → no surge."""
        result = volume_surge_check(None, "TEST")
        assert result.surged is False


# ─── Step 4: Two-day Confirmation Tests ──────────────────────────────────────

class TestConfirmation:
    """Test the Two-day Confirmation check (step 4)."""

    def test_confirmed_two_day_surge(self):
        """Both today and yesterday surged → confirmed=True, type='two_day_surge'."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=50_000)
        # Make last 2 days have surging volume
        df.iloc[-1, df.columns.get_loc("Volume")] = 200_000  # today
        df.iloc[-2, df.columns.get_loc("Volume")] = 180_000  # yesterday
        # Prior 3 days for today: -2(180k), -3(50k), -4(50k) → 200k > all? No, 200k > 180k? Yes
        # Prior 3 days for yesterday: -3(50k), -4(50k), -5(50k) → 180k > all? Yes
        # Wait, need to be more careful. Let me set up volumes properly.
        # Days: -5=50k, -4=50k, -3=50k, -2=180k (yesterday), -1=200k (today)
        # Today's prior 3: -2(180k), -3(50k), -4(50k) → 200k > 180k? Yes. 200k > 50k? Yes. 200k > 50k? Yes → surged
        # Yesterday's prior 3: -3(50k), -4(50k), -5(50k) → 180k > 50k? Yes. 180k > 50k? Yes. 180k > 50k? Yes → surged
        result = confirmation_check(df, "TEST")
        assert result.confirmed is True
        assert result.confirmation_type == "two_day_surge"

    def test_confirmed_pullback_after_strong(self):
        """Pullback after strong up day → confirmed=True, type='pullback_after_strong'."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=50_000)
        # Day before yesterday: price=48, yesterday: price=50 (+4.17%), today: price=49.5 (-1%)
        df.iloc[-3, df.columns.get_loc("Close")] = 48.0
        df.iloc[-2, df.columns.get_loc("Close")] = 50.0  # +4.17% → strong up day
        df.iloc[-1, df.columns.get_loc("Close")] = 49.5  # -1.0% → light pullback
        result = confirmation_check(df, "TEST")
        assert result.confirmed is True
        assert result.confirmation_type == "pullback_after_strong"

    def test_not_confirmed_first_day_spike(self):
        """Surge today only, no yesterday confirmation, no pullback → confirmed=False."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=50_000)
        # Only today has high volume, yesterday was normal
        df.iloc[-1, df.columns.get_loc("Volume")] = 200_000
        # No pullback pattern (prices are flat)
        result = confirmation_check(df, "TEST")
        assert result.confirmed is False
        assert result.confirmation_type == "none"

    def test_not_confirmed_insufficient_data(self):
        """Less than 5 days → not confirmed."""
        df = make_flat_ohlcv(days=3, price=50.0, volume=50_000)
        result = confirmation_check(df, "TEST")
        assert result.confirmed is False
        assert "غير كافية" in result.reason_ar

    def test_pullback_requires_strong_up_day(self):
        """Pullback after a weak day (< 3%) → not confirmed via pullback."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=50_000)
        df.iloc[-3, df.columns.get_loc("Close")] = 49.5
        df.iloc[-2, df.columns.get_loc("Close")] = 50.0  # +1.0% → not strong enough
        df.iloc[-1, df.columns.get_loc("Close")] = 49.7  # -0.6% → pullback
        result = confirmation_check(df, "TEST")
        # No surge, pullback too weak → not confirmed
        assert result.confirmed is False

    def test_pullback_rejects_large_decline(self):
        """Decline > 2% is not a 'light pullback' → not confirmed."""
        df = make_flat_ohlcv(days=10, price=50.0, volume=50_000)
        df.iloc[-3, df.columns.get_loc("Close")] = 47.0
        df.iloc[-2, df.columns.get_loc("Close")] = 50.0  # +6.4% → strong
        df.iloc[-1, df.columns.get_loc("Close")] = 48.5  # -3.0% → too large for pullback
        result = confirmation_check(df, "TEST")
        # The pullback is -3% which is < -2%, so not a light pullback
        assert result.confirmed is False


# ─── Step 5: Trend & Relative Strength Filter Tests ──────────────────────────

class TestTrendStrengthFilter:
    """Test the Trend & RS filter (step 5)."""

    def _make_df_with_trend(self, days=25, start_price=45.0, end_price=52.0):
        """Make a DataFrame where price goes from start to end (uptrend)."""
        dates = pd.bdate_range(start="2026-01-01", periods=days)
        prices = np.linspace(start_price, end_price, days)
        return pd.DataFrame({
            "Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
            "Close": prices, "Volume": [100_000] * days,
        }, index=dates)

    def test_passes_with_uptrend_and_rs(self):
        """Price > EMA50, RSI < 70, stock change > EGX 30 → pass."""
        df = self._make_df_with_trend(start_price=45.0, end_price=52.0)
        result = trend_strength_filter(
            df, ema50=48.0, rsi=55.0, egx30_change_20d=2.0, ticker="TEST")
        assert result.passed is True
        assert "اتجاه صاعد" in result.reason

    def test_fails_price_below_ema50(self):
        """Price < EMA50 → fail."""
        df = self._make_df_with_trend(start_price=45.0, end_price=46.0)
        result = trend_strength_filter(
            df, ema50=50.0, rsi=55.0, egx30_change_20d=2.0, ticker="TEST")
        assert result.passed is False
        assert "تحت EMA50" in result.reason

    def test_fails_rsi_overbought(self):
        """RSI >= 70 → fail."""
        df = self._make_df_with_trend(start_price=45.0, end_price=52.0)
        result = trend_strength_filter(
            df, ema50=48.0, rsi=75.0, egx30_change_20d=2.0, ticker="TEST")
        assert result.passed is False
        assert "تشبع شرائي" in result.reason

    def test_fails_weak_relative_strength(self):
        """Stock change < EGX 30 change → fail."""
        df = self._make_df_with_trend(start_price=50.0, end_price=51.0)  # +2%
        result = trend_strength_filter(
            df, ema50=49.0, rsi=55.0, egx30_change_20d=5.0, ticker="TEST")
        assert result.passed is False
        assert "قوة نسبية ضعيفة" in result.reason

    def test_fails_multiple_conditions(self):
        """Multiple failures should list all in reason."""
        df = self._make_df_with_trend(start_price=50.0, end_price=49.0)  # -2%
        result = trend_strength_filter(
            df, ema50=52.0, rsi=75.0, egx30_change_20d=5.0, ticker="TEST")
        assert result.passed is False
        assert "EMA50" in result.reason
        assert "تشبع" in result.reason
        assert "قوة نسبية" in result.reason

    def test_fails_insufficient_data(self):
        """< 20 days → fail."""
        df = self._make_df_with_trend(days=10)
        result = trend_strength_filter(
            df, ema50=48.0, rsi=55.0, egx30_change_20d=2.0, ticker="TEST")
        assert result.passed is False
        assert "غير كافية" in result.reason

    def test_exclusion_code_set_on_failure(self):
        """Failed trend/RS should have the correct exclusion code."""
        df = self._make_df_with_trend(start_price=50.0, end_price=49.0)
        result = trend_strength_filter(
            df, ema50=52.0, rsi=55.0, egx30_change_20d=5.0, ticker="TEST")
        assert result.exclusion_code == "اتجاه نازل أو قوة نسبية ضعيفة"


# ─── Step 6: Risk Management Filter Tests ────────────────────────────────────

class TestRiskFilter:
    """Test the Risk Management filter (step 6)."""

    def test_passes_with_good_rr(self):
        """R/R >= 2:1 → pass."""
        # price=100, ATR=2, stop=100-3=97, resistance=106
        # risk=3, reward=6, R/R=2.0
        result = risk_filter(current_price=100.0, atr=2.0, resistance=106.0, ticker="TEST")
        assert result.passed is True
        assert result.details["rr_ratio"] >= 2.0

    def test_fails_with_poor_rr(self):
        """R/R < 2:1 → fail."""
        # price=100, ATR=2, stop=97, resistance=100.5
        # risk=3, reward=0.5, R/R=0.17
        result = risk_filter(current_price=100.0, atr=2.0, resistance=100.5, ticker="TEST")
        assert result.passed is False
        assert "risk-reward غير كافٍ" in result.reason

    def test_fails_no_resistance(self):
        """No resistance above price → fail."""
        result = risk_filter(current_price=100.0, atr=2.0, resistance=0.0, ticker="TEST")
        assert result.passed is False
        assert "لا توجد مقاومة" in result.reason

    def test_fails_resistance_below_price(self):
        """Resistance below current price → fail."""
        result = risk_filter(current_price=100.0, atr=2.0, resistance=95.0, ticker="TEST")
        assert result.passed is False

    def test_fails_no_atr(self):
        """ATR=0 → fail."""
        result = risk_filter(current_price=100.0, atr=0.0, resistance=110.0, ticker="TEST")
        assert result.passed is False
        assert "ATR غير متاح" in result.reason

    def test_fails_no_price(self):
        """Price=0 → fail."""
        result = risk_filter(current_price=0.0, atr=2.0, resistance=110.0, ticker="TEST")
        assert result.passed is False

    def test_stop_loss_calculated_correctly(self):
        """Stop-loss = price - (1.5 × ATR)."""
        result = risk_filter(current_price=100.0, atr=2.0, resistance=110.0, ticker="TEST")
        # stop = 100 - (1.5 * 2) = 97
        assert result.details["stop_loss"] == 97.0

    def test_rr_ratio_calculated_correctly(self):
        """R/R = (resistance - price) / (price - stop_loss)."""
        # price=100, ATR=3, stop=100-4.5=95.5, resistance=110
        # risk=4.5, reward=10, R/R=10/4.5=2.22
        result = risk_filter(current_price=100.0, atr=3.0, resistance=110.0, ticker="TEST")
        assert abs(result.details["rr_ratio"] - (10.0 / 4.5)) < 0.01

    def test_excellent_rr_passes(self):
        """R/R >= 4:1 → pass with high score."""
        # price=100, ATR=1, stop=98.5, resistance=110
        # risk=1.5, reward=10, R/R=6.67
        result = risk_filter(current_price=100.0, atr=1.0, resistance=110.0, ticker="TEST")
        assert result.passed is True
        assert result.details["rr_ratio"] >= 4.0

    def test_exclusion_code_on_failure(self):
        """Failed risk filter should have correct exclusion code."""
        result = risk_filter(current_price=100.0, atr=2.0, resistance=100.5, ticker="TEST")
        assert result.exclusion_code == "risk-reward غير كافٍ"


# ─── Step 7: Scoring Engine v2 Tests ─────────────────────────────────────────

class TestScoringV2:
    """Test the new Liquidity-First v2 scoring engine."""

    def _make_analysis(self, price=50.0, ema50=48.0, rsi=55.0):
        """Create a minimal StockAnalysis for scoring."""
        from indicators import StockAnalysis, IndicatorResult
        return StockAnalysis(
            ticker="TEST", name="Test", name_ar="اختبار",
            current_price=price, daily_change_pct=1.5, volume=100_000,
            indicators=[
                IndicatorResult("EMA 50", "EMA 50", ema50, 1 if price > ema50 else -1, "صاعد", ""),
                IndicatorResult("RSI", "RSI", rsi, 1 if 40 < rsi < 70 else 0, "محايد", ""),
                IndicatorResult("ATR", "ATR", 2.0, 0, "محايد", ""),
            ],
        )

    def _make_confirmation(self, confirmed=True, conf_type="two_day_surge"):
        """Create a ConfirmationResult for testing."""
        return ConfirmationResult(
            confirmed=confirmed, confirmation_type=conf_type,
            reason_ar="تأكيد" if confirmed else "لا تأكيد",
            reason_en="confirmed" if confirmed else "not confirmed",
        )

    def test_score_in_valid_range(self):
        """Score should be 0-100."""
        analysis = self._make_analysis()
        conf = self._make_confirmation()
        result = compute_score_v2(
            analysis, avg_turnover_egp=50_000_000, latest_turnover_egp=55_000_000,
            stock_change_20d=8.0, egx30_change_20d=3.0,
            confirmation=conf, rr_ratio=2.5,
            stop_loss=47.0, target=56.0,
        )
        assert 0 <= result.total_score <= 100

    def test_weights_sum_to_100(self):
        """Factor weights must sum to 100."""
        assert sum(WEIGHTS.values()) == 100

    def test_strong_stock_gets_buy(self):
        """Stock with strong liquidity, trend, confirmation, and R/R → Buy."""
        analysis = self._make_analysis(price=52, ema50=48, rsi=58)
        conf = self._make_confirmation(confirmed=True, conf_type="two_day_surge")
        result = compute_score_v2(
            analysis, avg_turnover_egp=100_000_000, latest_turnover_egp=120_000_000,
            stock_change_20d=12.0, egx30_change_20d=3.0,
            confirmation=conf, rr_ratio=3.5,
            stop_loss=47.0, target=62.0,
        )
        assert result.recommendation == "Buy"
        assert result.total_score >= 70

    def test_weak_stock_gets_watch_or_lower(self):
        """Stock with weak factors → Watch or No Trade."""
        analysis = self._make_analysis(price=49, ema50=50, rsi=42)
        conf = self._make_confirmation(confirmed=False, conf_type="none")
        result = compute_score_v2(
            analysis, avg_turnover_egp=1_200_000, latest_turnover_egp=1_100_000,
            stock_change_20d=0.5, egx30_change_20d=0.1,
            confirmation=conf, rr_ratio=2.1,
            stop_loss=46.0, target=52.0,
        )
        assert result.total_score < 70  # Not a Buy

    def test_stop_loss_and_target_stored(self):
        """ScoringResult should store stop_loss and target."""
        analysis = self._make_analysis()
        conf = self._make_confirmation()
        result = compute_score_v2(
            analysis, avg_turnover_egp=50_000_000, latest_turnover_egp=55_000_000,
            stock_change_20d=8.0, egx30_change_20d=3.0,
            confirmation=conf, rr_ratio=2.5,
            stop_loss=47.0, target=56.0,
        )
        assert result.stop_loss == 47.0
        assert result.target == 56.0
        assert result.rr_ratio == 2.5

    def test_four_factors_present(self):
        """v2 scoring should have exactly 4 factors."""
        analysis = self._make_analysis()
        conf = self._make_confirmation()
        result = compute_score_v2(
            analysis, avg_turnover_egp=50_000_000, latest_turnover_egp=55_000_000,
            stock_change_20d=8.0, egx30_change_20d=3.0,
            confirmation=conf, rr_ratio=2.5,
            stop_loss=47.0, target=56.0,
        )
        factor_names = [f.name for f in result.factors]
        assert len(result.factors) == 4
        assert "Liquidity Strength" in factor_names
        assert "Trend & RS" in factor_names
        assert "Confirmation" in factor_names
        assert "Risk/Reward" in factor_names

    def test_stale_data_forces_no_trade(self):
        """Stale data → No Trade regardless of score."""
        analysis = self._make_analysis()
        conf = self._make_confirmation()
        result = compute_score_v2(
            analysis, avg_turnover_egp=100_000_000, latest_turnover_egp=120_000_000,
            stock_change_20d=12.0, egx30_change_20d=3.0,
            confirmation=conf, rr_ratio=3.5,
            stop_loss=47.0, target=62.0,
            data_freshness="stale",
        )
        assert result.recommendation == "No Trade"

    def test_low_data_quality_forces_no_trade(self):
        """Data quality < 0.7 → No Trade."""
        analysis = self._make_analysis()
        conf = self._make_confirmation()
        result = compute_score_v2(
            analysis, avg_turnover_egp=100_000_000, latest_turnover_egp=120_000_000,
            stock_change_20d=12.0, egx30_change_20d=3.0,
            confirmation=conf, rr_ratio=3.5,
            stop_loss=47.0, target=62.0,
            data_quality=0.5,
        )
        assert result.recommendation == "No Trade"
