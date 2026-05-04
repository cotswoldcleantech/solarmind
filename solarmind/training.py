"""Training module — PPO agent training on SolarMindEnv.

Uses Stable Baselines3 PPO with deterministic seeding.
Reproduces the prototype results documented in the Q9 appendix:
  - 150,000 timesteps (~105 episodes of 30-day horizon)
  - seed=42
  - Default PPO hyperparameters
  - Trained model converges by ~episode 40
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from solarmind.environment import SolarMindEnv
from solarmind.data import HouseholdProfile


def train_ppo(
    profile: Optional[HouseholdProfile] = None,
    total_timesteps: int = 150_000,
    seed: int = 42,
    output_dir: str = "models",
    verbose: int = 1,
):
    """Train a PPO agent on SolarMindEnv.

    Args:
        profile: Household configuration. Defaults to standard.
        total_timesteps: Number of environment steps.
        seed: Random seed for reproducibility.
        output_dir: Where to save the trained model.
        verbose: SB3 verbosity (0 = silent, 1 = info).

    Returns:
        Trained PPO model.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 is required for training. "
            "Install with: pip install stable-baselines3"
        ) from e

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Wrap env with Monitor for episode reward tracking
    env = Monitor(SolarMindEnv(profile=profile, seed=seed))

    # Standard PPO with default hyperparameters - matches Q9 appendix configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    if verbose:
        print(f"Training PPO for {total_timesteps:,} timesteps (seed={seed})...")

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / f"solarmind_ppo_seed{seed}"
    model.save(str(model_path))

    if verbose:
        print(f"Model saved to {model_path}.zip")

    return model
