# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-04

### Added
- Initial public prototype release
- `SolarMindEnv` Gymnasium environment for residential solar + battery control
- Synthetic household profile generator (solar, load, Octopus Agile-style tariff)
- `RuleBasedDispatcher` baseline (solar-first, tariff-blind)
- PPO training pipeline using Stable Baselines3
- Evaluation pipeline with 4 validation metrics matching the Q9 appendix
- Plotting utilities reproducing the figures from the Q9 appendix
- CLI: `solarmind train`, `evaluate`, `plot`, `demo`
- Unit tests covering environment, data, baseline, evaluation
- Apache 2.0 license
- README with full methodology, architecture, and roadmap

### Notes
- This is a single-household prototype. Results documented are reproducible
  with seed=42 and 150,000 training timesteps.
- Phase 1 work (extending to 20 households, EV asset, multi-objective reward,
  transfer learning) is funding-dependent.
