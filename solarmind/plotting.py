"""Plotting module — reproduces the figures from our Innovate UK Q9 appendix.

  - Figure 2: Training convergence curve
  - Figure 3: 4-metric comparison bar chart (PPO vs baseline)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional


def plot_training_curve(
    rewards: list[float],
    baseline_reward: float,
    output_path: str = "training_curve.png",
) -> str:
    """Plot training reward over episodes.

    Args:
        rewards: Per-episode reward list (typically pulled from Monitor wrapper).
        baseline_reward: Rule-based baseline mean reward (for the dashed line).
        output_path: Where to save the figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    rewards_arr = np.array(rewards)
    episodes = np.arange(1, len(rewards_arr) + 1)
    rolling = np.convolve(rewards_arr, np.ones(10) / 10, mode="valid")
    rolling_x = np.arange(10, len(rewards_arr) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
    ax.plot(episodes, rewards_arr, color="#52B788", alpha=0.4, linewidth=0.8,
            label="Episode reward")
    ax.plot(rolling_x, rolling, color="#0F2A1D", linewidth=2.0,
            label="10-episode rolling mean")
    ax.axhline(baseline_reward, color="#B86E2A", linestyle="--", linewidth=1.5,
               label=f"Rule-based baseline (£{baseline_reward:.1f})")

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (£)")
    ax.set_title("SolarMind PPO agent training convergence",
                 fontweight="bold", color="#0F2A1D")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_comparison(
    ppo_results: dict,
    baseline_results: dict,
    output_path: str = "comparison.png",
) -> str:
    """Plot 4-metric comparison: PPO vs rule-based baseline.

    Args:
        ppo_results: Output of evaluation.evaluate(ppo_model)
        baseline_results: Output of evaluation.evaluate(rule_based)
        output_path: Save path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    metrics = [
        ("Net 30-day\nelectricity bill (£)",
         baseline_results["net_30day_benefit_pounds"],
         ppo_results["net_30day_benefit_pounds"]),
        ("Export\nrevenue (£)",
         baseline_results["export_revenue_pounds"],
         ppo_results["export_revenue_pounds"]),
        ("Self-consumption\nrate (%)",
         baseline_results["self_consumption_rate_pct"],
         ppo_results["self_consumption_rate_pct"]),
        ("Total grid\nimport (kWh)",
         baseline_results["total_import_kwh"],
         ppo_results["total_import_kwh"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.0), dpi=150)
    for ax, (label, base, ppo) in zip(axes, metrics):
        bars = ax.bar(["Rule-based", "SolarMind\nPPO"], [base, ppo],
                      color=["#D8F3DC", "#2D6A4F"], edgecolor="#0F2A1D", linewidth=0.5)
        ax.set_title(label, fontsize=10, fontweight="bold", color="#0F2A1D")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        # Value labels
        for bar, val in zip(bars, [base, ppo]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#0F2A1D")

    fig.suptitle("SolarMind vs rule-based baseline | 30-day single-household simulation",
                 fontsize=11, fontweight="bold", color="#0F2A1D", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
