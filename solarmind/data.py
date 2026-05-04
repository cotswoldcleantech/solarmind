"""Synthetic household profile generator.

Produces realistic 30-min resolution time series for a UK household:
  - Solar PV generation (bell curve peaked at midday, modulated by weather)
  - Household electricity consumption (morning + evening peaks)
  - Octopus Agile-style time-of-use tariff (-8p to 54p/kWh)

These are *single-household synthetic profiles* used for the prototype.
Phase 1 will replace this with VAE-generated diversity across 20 households,
trained on UK Power Networks and Pecan Street open datasets.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HouseholdProfile:
    """Configuration for a synthetic household."""

    solar_kwp: float = 4.0  # PV array peak rating
    battery_kwh: float = 8.0  # Battery usable capacity
    daily_consumption_kwh: float = 8.0  # Average daily household electricity use
    weather_factor: float = 0.85  # 0.4 = cloudy, 0.85 = mixed, 1.05 = sunny
    timestep_minutes: int = 30  # Decision cadence
    days: int = 30  # Episode length


def solar_generation(t_steps: np.ndarray, profile: HouseholdProfile) -> np.ndarray:
    """Solar generation in kW at each timestep.

    Bell curve peaked at noon, with weather-driven scaling and small noise.
    """
    steps_per_day = 24 * 60 // profile.timestep_minutes
    hours_of_day = (t_steps % steps_per_day) * (profile.timestep_minutes / 60.0)
    # Bell curve, daylight 6am-8pm
    peak_h = 12.5
    sigma = 3.0
    base = np.exp(-((hours_of_day - peak_h) ** 2) / (2 * sigma ** 2))
    daylight = (hours_of_day >= 6.0) & (hours_of_day <= 20.0)
    base = base * daylight
    # Small daily weather variation (deterministic via seed elsewhere)
    rng = np.random.default_rng(42)
    daily_factor = 1.0 + 0.15 * rng.standard_normal(profile.days)
    daily_idx = (t_steps // steps_per_day).astype(int)
    daily_idx = np.clip(daily_idx, 0, profile.days - 1)
    weather = profile.weather_factor * daily_factor[daily_idx]
    return base * profile.solar_kwp * weather


def household_load(t_steps: np.ndarray, profile: HouseholdProfile) -> np.ndarray:
    """Household electricity consumption in kW at each timestep.

    Typical UK pattern: morning peak (7-9am), evening peak (5-9pm).
    """
    steps_per_day = 24 * 60 // profile.timestep_minutes
    hours_of_day = (t_steps % steps_per_day) * (profile.timestep_minutes / 60.0)
    factor = np.full_like(hours_of_day, 0.4)
    factor = np.where((hours_of_day >= 7) & (hours_of_day <= 9), 1.6, factor)
    factor = np.where((hours_of_day >= 12) & (hours_of_day <= 14), 1.0, factor)
    factor = np.where((hours_of_day >= 17) & (hours_of_day <= 21), 2.0, factor)
    factor = np.where((hours_of_day >= 22) | (hours_of_day < 6), 0.3, factor)
    base_kw = profile.daily_consumption_kwh / 24.0 / 0.7
    return base_kw * factor


def grid_price(t_steps: np.ndarray, profile: HouseholdProfile) -> np.ndarray:
    """Octopus Agile-style time-of-use tariff in p/kWh.

    Range: ~-8p (overnight oversupply) to ~54p (evening peak).
    """
    steps_per_day = 24 * 60 // profile.timestep_minutes
    hours_of_day = (t_steps % steps_per_day) * (profile.timestep_minutes / 60.0)

    p = np.zeros_like(hours_of_day)
    # Overnight: -3p to +13p
    overnight = hours_of_day < 3
    p = np.where(overnight, -3 + np.sin(hours_of_day) * 2, p)
    # Pre-dawn: 5p to 14p
    predawn = (hours_of_day >= 3) & (hours_of_day < 6)
    p = np.where(predawn, 8 + np.sin(hours_of_day * 2) * 3, p)
    # Morning: 18p to 26p
    morning = (hours_of_day >= 6) & (hours_of_day < 11)
    p = np.where(morning, 22 + np.sin(hours_of_day) * 4, p)
    # Midday solar dip: 9p to 15p
    midday = (hours_of_day >= 11) & (hours_of_day < 14)
    p = np.where(midday, 12 + np.sin(hours_of_day * 2) * 3, p)
    # Afternoon: 20p to 28p
    afternoon = (hours_of_day >= 14) & (hours_of_day < 16)
    p = np.where(afternoon, 24 + np.sin(hours_of_day) * 4, p)
    # Peak hours: 44p to 54p
    peak = (hours_of_day >= 16) & (hours_of_day < 19)
    p = np.where(peak, 48 + np.sin(hours_of_day * 3) * 4, p)
    # Late evening: 24p to 32p
    late = (hours_of_day >= 19) & (hours_of_day < 22)
    p = np.where(late, 28 + np.sin(hours_of_day * 2) * 4, p)
    # Pre-midnight: 11p to 17p
    pre_mid = hours_of_day >= 22
    p = np.where(pre_mid, 14 + np.sin(hours_of_day) * 3, p)

    return p


def generate_episode(profile: HouseholdProfile) -> dict:
    """Generate full episode time series for an episode."""
    steps_per_day = 24 * 60 // profile.timestep_minutes
    total_steps = profile.days * steps_per_day
    t = np.arange(total_steps)

    return {
        "t_steps": t,
        "solar_kw": solar_generation(t, profile),
        "load_kw": household_load(t, profile),
        "price_p_per_kwh": grid_price(t, profile),
        "steps_per_day": steps_per_day,
        "total_steps": total_steps,
        "profile": profile,
    }
