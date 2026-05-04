"""SolarMind Gymnasium environment.

A residential energy management environment where the agent controls:
  - Battery action (charge / discharge / hold)
  - Solar dispatch (use locally vs export)

Observation includes solar generation, household load, battery state,
grid price, and time-of-day features.

Reward is the per-step net monetary benefit (export revenue minus grid import cost).

Phase 1 will extend this to:
  - 3-asset action space (add EV charging)
  - Multi-objective reward (bill, export, self-consumption)
  - 20-household synthetic environment
"""

from __future__ import annotations
from typing import Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from solarmind.data import HouseholdProfile, generate_episode


class SolarMindEnv(gym.Env):
    """Residential energy management environment for reinforcement learning.

    Args:
        profile: Household configuration. If None, defaults are used.
        seed: Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human"], "render_fps": 0}

    def __init__(
        self,
        profile: Optional[HouseholdProfile] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.profile = profile or HouseholdProfile()
        self._np_random = np.random.default_rng(seed)

        # Continuous action: battery action in [-1, 1] (negative = discharge, positive = charge)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: 8-dim continuous
        # [solar_kw, load_kw, battery_soc_frac, price_p_per_kwh,
        #  hour_sin, hour_cos, day_of_week_sin, day_of_week_cos]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -10, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([10, 10, 1, 100, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode data populated on reset
        self._episode_data: Optional[dict] = None
        self._step_idx: int = 0
        self._battery_soc_kwh: float = 0.0

        # Battery physical constants
        self.battery_max_charge_rate_kw: float = 3.0  # Typical home battery
        self.battery_min_soc_frac: float = 0.10
        self.battery_max_soc_frac: float = 0.95
        self.battery_efficiency: float = 0.95

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._episode_data = generate_episode(self.profile)
        self._step_idx = 0
        # Start at 50% state of charge
        self._battery_soc_kwh = self.profile.battery_kwh * 0.5

        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._episode_data is None:
            raise RuntimeError("Must call reset() before step()")

        action_val = float(np.clip(action[0], -1.0, 1.0))

        # Current snapshot
        solar_kw = self._episode_data["solar_kw"][self._step_idx]
        load_kw = self._episode_data["load_kw"][self._step_idx]
        price_p = self._episode_data["price_p_per_kwh"][self._step_idx]

        # Time step in hours
        dt_h = self.profile.timestep_minutes / 60.0

        # Convert action to battery power (kW). Positive = charging, negative = discharging.
        battery_kw = action_val * self.battery_max_charge_rate_kw

        # Constrain by SoC bounds
        if battery_kw > 0:  # charging
            max_charge = (self.profile.battery_kwh * self.battery_max_soc_frac
                          - self._battery_soc_kwh) / dt_h
            battery_kw = min(battery_kw, max_charge)
        else:  # discharging
            max_discharge = (self._battery_soc_kwh
                             - self.profile.battery_kwh * self.battery_min_soc_frac) / dt_h
            battery_kw = max(battery_kw, -max_discharge)

        # Energy flow this step
        # Solar covers load first, then battery, then export
        # Battery action is whatever the agent chose (within physical limits)
        # Net grid flow = load + battery_kw - solar
        # Positive = import, negative = export
        net_grid_kw = load_kw + battery_kw - solar_kw

        if net_grid_kw >= 0:
            # Importing from grid
            cost_p = net_grid_kw * dt_h * price_p
            export_kwh = 0.0
            import_kwh = net_grid_kw * dt_h
        else:
            # Exporting to grid
            cost_p = net_grid_kw * dt_h * price_p  # negative = revenue
            export_kwh = -net_grid_kw * dt_h
            import_kwh = 0.0

        # Update battery SoC
        if battery_kw > 0:
            self._battery_soc_kwh += battery_kw * dt_h * self.battery_efficiency
        else:
            self._battery_soc_kwh += battery_kw * dt_h / self.battery_efficiency

        self._battery_soc_kwh = float(np.clip(
            self._battery_soc_kwh,
            self.profile.battery_kwh * self.battery_min_soc_frac,
            self.profile.battery_kwh * self.battery_max_soc_frac,
        ))

        # Reward: negative cost, in £ per step
        reward_pounds = -cost_p / 100.0

        # Advance time
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self._episode_data["total_steps"]

        info = self._info()
        info.update({
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "cost_pounds": cost_p / 100.0,
            "battery_kw": battery_kw,
            "solar_kw": solar_kw,
            "load_kw": load_kw,
            "price_p_per_kwh": price_p,
        })

        return self._observation(), reward_pounds, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        if self._step_idx >= self._episode_data["total_steps"]:
            idx = self._episode_data["total_steps"] - 1
        else:
            idx = self._step_idx

        steps_per_day = self._episode_data["steps_per_day"]
        hour_of_day = (idx % steps_per_day) * (self.profile.timestep_minutes / 60.0)
        day_of_episode = idx // steps_per_day

        return np.array([
            self._episode_data["solar_kw"][idx],
            self._episode_data["load_kw"][idx],
            self._battery_soc_kwh / self.profile.battery_kwh,
            self._episode_data["price_p_per_kwh"][idx],
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24),
            np.sin(2 * np.pi * day_of_episode / 7),
            np.cos(2 * np.pi * day_of_episode / 7),
        ], dtype=np.float32)

    def _info(self) -> dict:
        return {
            "step": self._step_idx,
            "battery_soc_kwh": self._battery_soc_kwh,
            "battery_soc_frac": self._battery_soc_kwh / self.profile.battery_kwh,
        }

    def render(self) -> None:
        if self._episode_data is None or self._step_idx == 0:
            return
        idx = max(0, self._step_idx - 1)
        steps_per_day = self._episode_data["steps_per_day"]
        h = (idx % steps_per_day) * (self.profile.timestep_minutes / 60.0)
        print(
            f"Step {idx:4d}  Hour {h:5.1f}  "
            f"Solar {self._episode_data['solar_kw'][idx]:5.2f} kW  "
            f"Load {self._episode_data['load_kw'][idx]:5.2f} kW  "
            f"Price {self._episode_data['price_p_per_kwh'][idx]:5.1f}p  "
            f"Battery {self._battery_soc_kwh:5.2f} kWh"
        )
