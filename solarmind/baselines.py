"""Rule-based baseline dispatcher.

Implements the "industry standard" rule used by typical solar inverter controllers:
  - Solar covers load first
  - Surplus solar goes to battery (until full)
  - Surplus beyond battery is exported
  - Load deficit is covered first by battery (until at minimum SoC), then grid

This baseline is simple and effective for self-consumption maximisation,
but ignores grid price signals. It's the comparison point for the SolarMind PPO agent.

In Q9 of our Innovate UK application, this is described as the "rule-based dispatcher"
referenced in our preliminary results (achieves +£8.79 net 30-day bill in our prototype).
"""

from __future__ import annotations
import numpy as np


class RuleBasedDispatcher:
    """Solar-first, then battery, then grid: tariff-blind."""

    def __init__(self, battery_max_charge_kw: float = 3.0):
        self.battery_max_charge_kw = battery_max_charge_kw

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Compute action from observation.

        Returns the same (action, state) tuple as Stable Baselines3 PPO.predict().
        State is unused (returned as None) for compatibility.
        """
        solar_kw = observation[0]
        load_kw = observation[1]
        # battery_soc_frac = observation[2]
        # price_p_per_kwh = observation[3]  # ← deliberately ignored

        surplus = solar_kw - load_kw
        if surplus > 0:
            # Charge battery from surplus, scaled to max charge rate
            action = float(np.clip(surplus / self.battery_max_charge_kw, 0.0, 1.0))
        else:
            # Discharge battery to cover deficit
            deficit = -surplus
            action = float(np.clip(-deficit / self.battery_max_charge_kw, -1.0, 0.0))

        return np.array([action], dtype=np.float32), None
