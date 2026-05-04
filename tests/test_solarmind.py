"""Unit tests for the SolarMind package.

Run with:
    pytest tests/

These tests verify the environment, data generators, and baseline are
behaving correctly without requiring training (which is slow).
"""

import numpy as np
import pytest

from solarmind import SolarMindEnv, RuleBasedDispatcher
from solarmind.data import HouseholdProfile, generate_episode, solar_generation, household_load, grid_price


# ===== Environment tests =====


def test_env_reset():
    env = SolarMindEnv(seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (8,)
    assert info["step"] == 0
    assert 0 <= info["battery_soc_frac"] <= 1


def test_env_step():
    env = SolarMindEnv(seed=42)
    env.reset(seed=42)
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (8,)
    assert isinstance(reward, float)
    assert not terminated
    assert "import_kwh" in info
    assert "export_kwh" in info


def test_env_full_episode():
    """Run a full 30-day episode with a constant action."""
    env = SolarMindEnv(seed=42)
    env.reset(seed=42)
    n_steps = 0
    while True:
        action = np.array([0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        if terminated or truncated:
            break
    # 30 days × 48 steps/day (30-min granularity) = 1440 steps
    assert n_steps == 30 * 48


def test_env_action_clipping():
    env = SolarMindEnv(seed=42)
    env.reset(seed=42)
    # Way out-of-range action
    action = np.array([10.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    # Should not crash; battery should respect physical limits
    assert info["battery_soc_frac"] <= 0.95


def test_env_reproducibility():
    """Same seed produces same trajectory."""
    env1 = SolarMindEnv(seed=42)
    env2 = SolarMindEnv(seed=42)
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_almost_equal(obs1, obs2)

    for _ in range(10):
        action = np.array([0.3], dtype=np.float32)
        o1, r1, _, _, _ = env1.step(action)
        o2, r2, _, _, _ = env2.step(action)
        np.testing.assert_array_almost_equal(o1, o2)
        assert abs(r1 - r2) < 1e-6


# ===== Data tests =====


def test_solar_generation_zero_at_night():
    profile = HouseholdProfile()
    # Step 0 = midnight
    t = np.array([0])
    gen = solar_generation(t, profile)
    assert gen[0] == 0.0


def test_solar_generation_peak_at_midday():
    profile = HouseholdProfile(solar_kwp=4.0, weather_factor=1.0)
    # 30-min steps; midday is step 24
    t = np.array([0, 12, 24, 36])  # midnight, 6am, midday, 6pm
    gen = solar_generation(t, profile)
    # Midday should be the highest value of these
    assert gen[2] == max(gen)
    # Midnight zero
    assert gen[0] == 0


def test_household_load_evening_peak():
    profile = HouseholdProfile(daily_consumption_kwh=8.0)
    # Step 36 = 18:00 (evening peak)
    # Step 24 = 12:00 (midday)
    t = np.array([24, 36])
    load = household_load(t, profile)
    # Evening should be higher than midday
    assert load[1] > load[0]


def test_grid_price_evening_peak():
    profile = HouseholdProfile()
    # Evening peak should be higher than overnight
    t_overnight = np.array([4])  # 02:00
    t_peak = np.array([34])  # 17:00
    p_overnight = grid_price(t_overnight, profile)
    p_peak = grid_price(t_peak, profile)
    assert p_peak[0] > p_overnight[0]
    assert p_peak[0] > 30  # Peak should be at least 30p
    assert p_overnight[0] < 10  # Overnight should be < 10p


def test_grid_price_can_be_negative():
    profile = HouseholdProfile()
    # Around 02:00 the price formula goes negative
    t = np.array([2])  # 01:00
    p = grid_price(t, profile)
    assert p[0] < 5  # at least sub-5p


def test_episode_lengths():
    profile = HouseholdProfile(days=30, timestep_minutes=30)
    data = generate_episode(profile)
    expected = 30 * 24 * 60 // 30
    assert data["total_steps"] == expected
    assert len(data["solar_kw"]) == expected
    assert len(data["load_kw"]) == expected
    assert len(data["price_p_per_kwh"]) == expected


# ===== Baseline tests =====


def test_baseline_charges_on_surplus():
    """When solar > load, baseline should charge battery (positive action)."""
    baseline = RuleBasedDispatcher()
    obs = np.array([3.0, 0.5, 0.5, 20.0, 0, 0, 0, 0], dtype=np.float32)
    action, _ = baseline.predict(obs)
    assert action[0] > 0  # charging


def test_baseline_discharges_on_deficit():
    """When load > solar, baseline should discharge battery (negative action)."""
    baseline = RuleBasedDispatcher()
    obs = np.array([0.2, 2.0, 0.5, 30.0, 0, 0, 0, 0], dtype=np.float32)
    action, _ = baseline.predict(obs)
    assert action[0] < 0  # discharging


def test_baseline_ignores_price():
    """Baseline should produce same action regardless of grid price."""
    baseline = RuleBasedDispatcher()
    obs_cheap = np.array([1.0, 1.0, 0.5, -5.0, 0, 0, 0, 0], dtype=np.float32)
    obs_peak = np.array([1.0, 1.0, 0.5, 50.0, 0, 0, 0, 0], dtype=np.float32)
    action_cheap, _ = baseline.predict(obs_cheap)
    action_peak, _ = baseline.predict(obs_peak)
    np.testing.assert_array_equal(action_cheap, action_peak)


# ===== Evaluation pipeline test =====


def test_baseline_full_evaluation():
    """End-to-end: baseline runs through a full episode without errors."""
    from solarmind.evaluation import evaluate

    baseline = RuleBasedDispatcher()
    results = evaluate(baseline, seed=42)

    assert "net_30day_benefit_pounds" in results
    assert "self_consumption_rate_pct" in results
    assert "export_revenue_pounds" in results
    # Self-consumption should be reasonable (>30%, <100%)
    assert 30 < results["self_consumption_rate_pct"] < 100
