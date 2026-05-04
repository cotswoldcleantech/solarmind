# SolarMind

**Reinforcement learning for residential solar, battery and EV optimisation in the UK.**

A research prototype that uses Proximal Policy Optimisation (PPO) to autonomously
manage solar dispatch and battery storage decisions for UK households on dynamic
time-of-use tariffs (Octopus Agile-style pricing).

## Status

🟡 **Pre-funded research prototype.** Single-household synthetic environment;
solar + battery (no EV yet); single-objective reward. Phase 1 (Innovate UK
funding-dependent) will extend to 20 households, EV asset, multi-objective
reward, and synthetic-to-real transfer learning.

## What this prototype demonstrates

A trained PPO agent that learns to:
- Charge the battery from grid during cheap overnight hours (negative or low Octopus Agile prices)
- Discharge the battery during 4–7pm peak hours when grid prices reach 40–54p/kWh
- Export at peak times to maximise revenue
- Store solar surplus when the battery is empty and prices will rise

vs a rule-based baseline that:
- Solar covers load first (always)
- Surplus solar charges battery
- Battery discharges when load exceeds solar
- Price-blind

## Validated 30-day single-household results

The trained PPO agent significantly outperforms a rule-based baseline on
monetary metrics (net bill, export revenue) by exploiting Octopus Agile-style
tariff arbitrage. The baseline retains higher self-consumption rate because
its solar-first rule is well-suited to that specific objective.

**To reproduce on your own machine:**

```bash
solarmind train --timesteps 150000 --seed 42
solarmind evaluate --model models/solarmind_ppo_seed42
```

Run `solarmind evaluate` (without `--model`) to see baseline numbers immediately
on your synthetic data. Exact numbers depend on your installed library versions
and hardware; the qualitative ordering (PPO better on bill + export, baseline
better on self-consumption) is robust across environments.

⚠ **Honest interpretation of the trade-off:** The PPO agent maximises monetary
benefit, which under Octopus Agile-style tariffs rewards exporting at peak
prices rather than maximum self-consumption. It is correctly maximising its
reward function — the reward function itself needs revising to be
multi-objective. Phase 1 introduces a multi-objective reward combining bill,
export revenue, and self-consumption with tunable weights.

⚠ **Note on numbers vs the Q9 appendix:** The exact numerical results
documented in our Innovate UK Q9 appendix were produced by the original Colab
prototype, which uses a slightly different synthetic environment configuration.
This open-source repo produces qualitatively identical results
(PPO outperforms baseline on monetary metrics; baseline outperforms on
self-consumption) but the precise figures vary depending on the household
profile, weather settings, and library versions. The methodology and
direction of effects are the same; the absolute numbers should be
reproduced from this codebase, not assumed to match a prior figure.

## Quickstart

### Install

```bash
git clone https://github.com/cotswoldcleantech/solarmind.git
cd solarmind
pip install -e .
```

Optional dev tools:

```bash
pip install -e ".[dev]"
```

### Reproduce the prototype results

```bash
# Train PPO for full 150,000 timesteps (~10 minutes on a laptop)
solarmind train --timesteps 150000 --seed 42

# Evaluate the trained model and the rule-based baseline
solarmind evaluate --model models/solarmind_ppo_seed42

# Reproduce the comparison plot
solarmind plot --model models/solarmind_ppo_seed42 --output comparison.png
```

### Quick demo (short training)

```bash
solarmind demo --timesteps 10000
```

This trains for 10,000 timesteps (~1 minute) and runs the full evaluation pipeline.
Results will be weaker than the full 150,000-step run, but the structure is identical.

### Try in Google Colab

A Colab notebook reproducing the full results is in `examples/quickstart_colab.ipynb`.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            REAL-TIME INPUTS                         │
│  Solar gen │ Battery state │ Grid price │ Load     │
└──────────────────┬──────────────────────────────────┘
                   ▼
        ┌─────────────────────┐
        │   STATE ENCODER     │   8-dim observation
        │   30-min cadence    │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │  PPO RL AGENT       │   Stable Baselines3
        │  MlpPolicy          │   2-layer MLP
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ BATTERY ACTION      │   Continuous [-1, 1]
        │ (charge/discharge)  │
        └─────────────────────┘
```

### Observation space (8-dim continuous)

| Dim | Variable | Range |
|---|---|---|
| 0 | Solar generation (kW) | 0 to ~6 |
| 1 | Household load (kW) | 0 to ~3 |
| 2 | Battery state of charge (frac) | 0.10–0.95 |
| 3 | Grid price (p/kWh) | -8 to 54 |
| 4 | sin(2π · hour / 24) | -1 to 1 |
| 5 | cos(2π · hour / 24) | -1 to 1 |
| 6 | sin(2π · day / 7) | -1 to 1 |
| 7 | cos(2π · day / 7) | -1 to 1 |

### Action space (1-dim continuous)

Single action in [-1, 1]:
- Negative = discharge battery
- Zero = hold
- Positive = charge battery

Scaled by `battery_max_charge_rate_kw` (default 3.0 kW).

### Reward function

Per step: `reward = -cost_pounds`, where `cost_pounds = net_grid_kw × dt × price`.
Positive reward = customer made money (export); negative = customer paid (import).

### Methodology table

| Component | Implementation |
|---|---|
| RL algorithm | PPO (Stable Baselines3 v2.8.0) |
| Environment | Custom Gymnasium env, 30-day episode, 30-min step |
| Training | 150,000 timesteps (~105 episodes), seed=42 |
| Hyperparameters | Default SB3 PPO (3e-4 LR, 0.2 clip, 0.99 γ, 0.95 GAE λ) |
| Synthetic data | Single household profile, deterministic generators |
| Tariff | Octopus Agile-style: -8p to 54p/kWh, 30-min granularity |
| Baseline | Rule-based dispatcher: solar-first, then battery, then grid |

## Repo layout

```
solarmind/
├── solarmind/              # Main package
│   ├── __init__.py
│   ├── data.py             # Synthetic household profile generator
│   ├── environment.py      # Gymnasium environment (SolarMindEnv)
│   ├── baselines.py        # Rule-based baseline dispatcher
│   ├── training.py         # PPO training loop
│   ├── evaluation.py       # Evaluation + 4 validation metrics
│   ├── plotting.py         # Reproduces Q9 appendix figures
│   └── cli.py              # Command-line interface
├── tests/                  # Unit tests (pytest)
├── configs/                # YAML configuration examples
├── docs/                   # Documentation
├── examples/               # Colab notebook + sample scripts
├── pyproject.toml          # Package metadata + dependencies
├── LICENSE                 # Apache 2.0
└── README.md               # You are here
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests cover:
- Environment reset / step / action clipping / reproducibility
- Synthetic data generators (solar, load, price profiles)
- Rule-based baseline behaviour
- End-to-end evaluation pipeline

## Project context

This work was developed under [Innovate UK Application 10200004](https://www.gov.uk/government/organisations/innovate-uk)
(AI Champions: Frontier AI Phase 1) by Cotswold Cleantech Energy Ltd. The full
application — including methodology, validation plan, IP strategy, and Phase 2
roadmap — is private to the funding body but technical details are summarised
in this README.

The prototype is the technical evidence underpinning the application's
Q9 (technical development) and Q15 (risks) responses. It demonstrates:
- Feasibility of PPO convergence on the household energy environment
- The self-consumption / arbitrage trade-off that motivates Phase 1's multi-objective reward design
- Reproducibility (seed-fixed, end-to-end in <10 minutes)

## Roadmap

### Today (this prototype)
- ✅ Single synthetic household
- ✅ Two assets: solar + battery
- ✅ Single-objective reward (monetary)
- ✅ Rule-based baseline comparison
- ✅ Reproducible training pipeline

### Phase 1 (Innovate UK-dependent, 6 months)
- 🔲 20-household synthetic environment via VAE-generated diversity
- 🔲 Add EV charging as third controllable asset
- 🔲 Multi-objective reward (bill + export + self-consumption, tunable weights)
- 🔲 Synthetic-to-real transfer learning methodology
- 🔲 Two additional baselines: LSTM forecaster + non-adaptive PPO
- 🔲 Statistical validation: paired t-tests across 10 independent runs

### Phase 2 (Innovate UK Phase 2 + private funding-dependent)
- 🔲 Real-home demonstrator: 50–100 UK households
- 🔲 Inverter API integrations (SolarEdge, Fronius, Enphase)
- 🔲 Customer-facing mobile app + dashboard
- 🔲 Real-world validation against synthetic predictions

## License

Apache 2.0 — see [LICENSE](LICENSE).

This codebase uses standard open-source libraries:
- PyTorch (BSD)
- Stable Baselines3 (MIT)
- Gymnasium (MIT)

Training data is synthetic. No third-party data is included in this repository.

## Contact

**Cotswold Cleantech Energy Ltd**
Cheltenham, United Kingdom
Companies House No. 16962739
Email: info@ccenergy.uk

## Citation

If this work is useful in your research, please cite:

```bibtex
@misc{solarmind2026,
  title = {SolarMind: Reinforcement learning for residential solar, battery and EV optimisation},
  author = {Cotswold Cleantech Energy Ltd},
  year = {2026},
  url = {https://github.com/cotswold-cleantech/solarmind}
}
```

## Disclaimer

This is a research prototype, not a deployed product. It is not connected to
any real inverter, smart meter, or grid infrastructure. The synthetic data
used for training does not represent any specific real household. Results
should be treated as proof-of-concept, not as guidance for any real-world
energy management decision.
