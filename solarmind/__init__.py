"""SolarMind — Reinforcement learning for residential solar, battery and EV optimisation.

A reinforcement learning system that autonomously optimises energy decisions for
UK homeowners with rooftop solar, battery storage, and (in Phase 1) EV charging.

Cotswold Cleantech Energy Ltd
Innovate UK Application 10200004 — AI Champions: Frontier AI Phase 1
"""

__version__ = "0.1.0"
__author__ = "Cotswold Cleantech Energy Ltd"
__license__ = "Apache-2.0"

from solarmind.environment import SolarMindEnv
from solarmind.baselines import RuleBasedDispatcher

__all__ = ["SolarMindEnv", "RuleBasedDispatcher", "__version__"]
