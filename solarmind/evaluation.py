"""Evaluation module — run a trained model (or baseline) on the environment.

Computes the 4 validation metrics defined in our Innovate UK Q9 appendix:
  - Net 30-day electricity bill (£)
  - Grid export revenue (£)
  - Self-consumption rate (%)
  - Total grid import (kWh)
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

from solarmind.environment import SolarMindEnv
from solarmind.data import HouseholdProfile


def evaluate(
    agent: Any,
    profile: Optional[HouseholdProfile] = None,
    seed: int = 42,
    deterministic: bool = True,
) -> dict:
    """Run a single 30-day evaluation episode.

    Args:
        agent: Anything with a `.predict(obs, deterministic=...)` method that
               returns (action, state). Both PPO and RuleBasedDispatcher work.
        profile: Household profile. Defaults match Q9 appendix configuration.
        seed: Reproducibility seed.
        deterministic: For PPO, use deterministic policy.

    Returns:
        Dictionary with all 4 validation metrics + raw time series.
    """
    env = SolarMindEnv(profile=profile, seed=seed)
    obs, _ = env.reset(seed=seed)

    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_solar_kwh = 0.0
    total_load_kwh = 0.0
    total_cost_pounds = 0.0

    rewards = []
    log_solar, log_load, log_price, log_action, log_soc = [], [], [], [], []

    while True:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        total_import_kwh += info["import_kwh"]
        total_export_kwh += info["export_kwh"]
        total_solar_kwh += info["solar_kw"] * (env.profile.timestep_minutes / 60.0)
        total_load_kwh += info["load_kw"] * (env.profile.timestep_minutes / 60.0)
        total_cost_pounds += info["cost_pounds"]

        log_solar.append(info["solar_kw"])
        log_load.append(info["load_kw"])
        log_price.append(info["price_p_per_kwh"])
        log_action.append(info["battery_kw"])
        log_soc.append(info["battery_soc_frac"])

        if terminated or truncated:
            break

    # Self-consumption: fraction of generated solar that was used locally (not exported)
    self_consumed_kwh = total_solar_kwh - total_export_kwh
    self_consumption_rate = (
        100.0 * self_consumed_kwh / total_solar_kwh
        if total_solar_kwh > 0 else 0.0
    )

    # Net bill: positive = customer paid net, negative = customer received net
    # In our convention, positive cost_pounds is import, negative is export revenue.
    # So "net bill" the customer experiences = -total_cost_pounds (sign flip).
    # Wait - let me re-check: cost_pounds = net_grid_kw * dt * price.
    # net_grid_kw > 0 = importing; cost_pounds > 0 = customer paid.
    # net_grid_kw < 0 = exporting; cost_pounds < 0 = customer was paid.
    # So "net 30-day bill" in everyday terms = sum(cost_pounds).
    # Positive value = customer paid net (lost money to the grid).
    # Negative value = customer received net (made money on export).
    # In our prototype, the trained agent produces ~£-62 (made money), baseline ~£-9.
    # So we report: net_30day_benefit = -total_cost_pounds (negative cost = benefit)
    net_30day_benefit_pounds = -total_cost_pounds

    # Export revenue (£): when price was positive
    export_revenue_pounds = sum(
        max(0, info_kw) for info_kw, info_p in zip(
            [-info["battery_kw"] if False else 0 for info in []], []
        )
    )
    # Recompute export revenue properly from the log
    dt_h = env.profile.timestep_minutes / 60.0
    export_revenue_pounds = 0.0
    for i in range(len(log_solar)):
        net_grid_kw_i = log_load[i] + log_action[i] - log_solar[i]
        if net_grid_kw_i < 0:  # exporting
            export_revenue_pounds += -net_grid_kw_i * dt_h * log_price[i] / 100.0

    return {
        "net_30day_benefit_pounds": net_30day_benefit_pounds,
        "export_revenue_pounds": export_revenue_pounds,
        "self_consumption_rate_pct": self_consumption_rate,
        "total_import_kwh": total_import_kwh,
        "total_export_kwh": total_export_kwh,
        "total_solar_kwh": total_solar_kwh,
        "total_load_kwh": total_load_kwh,
        "total_reward": float(sum(rewards)),
        "n_steps": len(rewards),
        "time_series": {
            "solar_kw": np.array(log_solar),
            "load_kw": np.array(log_load),
            "price_p_per_kwh": np.array(log_price),
            "battery_kw": np.array(log_action),
            "battery_soc_frac": np.array(log_soc),
        },
    }


def print_results(name: str, results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Net 30-day benefit:    £{results['net_30day_benefit_pounds']:>8.2f}")
    print(f"  Export revenue:        £{results['export_revenue_pounds']:>8.2f}")
    print(f"  Self-consumption rate: {results['self_consumption_rate_pct']:>8.1f} %")
    print(f"  Total import:          {results['total_import_kwh']:>8.1f} kWh")
    print(f"  Total export:          {results['total_export_kwh']:>8.1f} kWh")
    print(f"  Total solar generated: {results['total_solar_kwh']:>8.1f} kWh")
    print(f"  Episode reward sum:    £{results['total_reward']:>8.2f}")
