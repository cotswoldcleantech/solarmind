"""SolarMind command-line interface.

Usage:
  solarmind train       # Train a PPO agent (default: 150k timesteps)
  solarmind evaluate    # Evaluate a trained model + baseline
  solarmind plot        # Reproduce the figures from the Q9 appendix
  solarmind demo        # Quick end-to-end run (short training)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def cmd_train(args):
    from solarmind.training import train_ppo
    from solarmind.data import HouseholdProfile

    profile = HouseholdProfile(
        solar_kwp=args.solar_kwp,
        battery_kwh=args.battery_kwh,
        daily_consumption_kwh=args.daily_kwh,
    )

    train_ppo(
        profile=profile,
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir=args.output,
        verbose=1,
    )
    return 0


def cmd_evaluate(args):
    from solarmind.environment import SolarMindEnv
    from solarmind.baselines import RuleBasedDispatcher
    from solarmind.evaluation import evaluate, print_results
    from solarmind.data import HouseholdProfile

    profile = HouseholdProfile(
        solar_kwp=args.solar_kwp,
        battery_kwh=args.battery_kwh,
        daily_consumption_kwh=args.daily_kwh,
    )

    # Always evaluate rule-based baseline
    baseline = RuleBasedDispatcher()
    baseline_results = evaluate(baseline, profile=profile, seed=args.seed)
    print_results("Rule-based baseline", baseline_results)

    # If model path provided, evaluate PPO too
    if args.model:
        try:
            from stable_baselines3 import PPO
        except ImportError:
            print("ERROR: stable_baselines3 not installed. Cannot load PPO model.",
                  file=sys.stderr)
            return 1

        model_path = Path(args.model)
        if not model_path.exists() and not model_path.with_suffix(".zip").exists():
            print(f"ERROR: Model file not found: {args.model}", file=sys.stderr)
            return 1

        ppo = PPO.load(str(model_path))
        ppo_results = evaluate(ppo, profile=profile, seed=args.seed)
        print_results("SolarMind PPO agent", ppo_results)

        print(f"\n{'=' * 60}")
        print("  Comparison")
        print(f"{'=' * 60}")
        diff_bill = (ppo_results["net_30day_benefit_pounds"]
                     - baseline_results["net_30day_benefit_pounds"])
        diff_export = (ppo_results["export_revenue_pounds"]
                       - baseline_results["export_revenue_pounds"])
        print(f"  Net benefit difference: £{diff_bill:+.2f}")
        print(f"  Export revenue diff:    £{diff_export:+.2f}")

    return 0


def cmd_plot(args):
    from solarmind.environment import SolarMindEnv
    from solarmind.baselines import RuleBasedDispatcher
    from solarmind.evaluation import evaluate
    from solarmind.plotting import plot_comparison
    from solarmind.data import HouseholdProfile

    profile = HouseholdProfile(
        solar_kwp=args.solar_kwp,
        battery_kwh=args.battery_kwh,
        daily_consumption_kwh=args.daily_kwh,
    )

    print("Evaluating rule-based baseline...")
    baseline = RuleBasedDispatcher()
    baseline_results = evaluate(baseline, profile=profile, seed=args.seed)

    if not args.model:
        print("ERROR: --model required for comparison plot.", file=sys.stderr)
        return 1

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("ERROR: stable_baselines3 not installed.", file=sys.stderr)
        return 1

    print("Evaluating PPO agent...")
    ppo = PPO.load(args.model)
    ppo_results = evaluate(ppo, profile=profile, seed=args.seed)

    output_path = plot_comparison(ppo_results, baseline_results, args.output)
    print(f"Saved comparison figure to {output_path}")
    return 0


def cmd_demo(args):
    """End-to-end quick demo: train short, evaluate, compare."""
    from solarmind.training import train_ppo
    from solarmind.environment import SolarMindEnv
    from solarmind.baselines import RuleBasedDispatcher
    from solarmind.evaluation import evaluate, print_results

    print("=" * 60)
    print("  SolarMind end-to-end demo")
    print("=" * 60)
    print()
    print(f"Training PPO for {args.timesteps:,} timesteps (this is a short")
    print(f"demo; the full training in our Q9 appendix uses 150,000 timesteps).")
    print()

    model = train_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        output_dir="models",
        verbose=0,
    )

    print()
    print("Evaluating both agents on a 30-day episode...")

    baseline = RuleBasedDispatcher()
    baseline_results = evaluate(baseline, seed=args.seed)
    print_results("Rule-based baseline", baseline_results)

    ppo_results = evaluate(model, seed=args.seed)
    print_results("SolarMind PPO agent", ppo_results)

    print(f"\n{'=' * 60}")
    print("  NOTE")
    print(f"{'=' * 60}")
    print("  This is a short demo with reduced training. Full results")
    print("  reported in our Innovate UK Q9 appendix use 150,000 timesteps")
    print("  (~10 minutes on Google Colab T4).")
    print()
    print(f"  Run `solarmind train --timesteps 150000` for full reproduction.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="solarmind",
        description="SolarMind - RL for residential solar/battery/EV optimisation",
    )
    sub = parser.add_subparsers(dest="command")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    common.add_argument("--solar-kwp", type=float, default=4.0, help="Solar PV size (kWp)")
    common.add_argument("--battery-kwh", type=float, default=8.0, help="Battery capacity (kWh)")
    common.add_argument("--daily-kwh", type=float, default=8.0, help="Daily consumption (kWh)")

    # Train
    p_train = sub.add_parser("train", help="Train PPO agent", parents=[common])
    p_train.add_argument("--timesteps", type=int, default=150_000,
                         help="Total training timesteps (default: 150,000)")
    p_train.add_argument("--output", default="models", help="Output dir")
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model and/or baseline",
                            parents=[common])
    p_eval.add_argument("--model", help="Path to trained PPO model (optional)")
    p_eval.set_defaults(func=cmd_evaluate)

    # Plot
    p_plot = sub.add_parser("plot", help="Reproduce comparison plot from Q9 appendix",
                            parents=[common])
    p_plot.add_argument("--model", required=True, help="Path to trained PPO model")
    p_plot.add_argument("--output", default="comparison.png", help="Output filename")
    p_plot.set_defaults(func=cmd_plot)

    # Demo
    p_demo = sub.add_parser("demo", help="Quick end-to-end demo (short training)",
                            parents=[common])
    p_demo.add_argument("--timesteps", type=int, default=10_000,
                        help="Demo training timesteps (default: 10,000)")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
