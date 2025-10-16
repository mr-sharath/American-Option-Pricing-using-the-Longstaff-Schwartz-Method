"""
Simulation module for stock price path generation.

This module provides different methods for simulating stock price paths
under Geometric Brownian Motion (GBM).
"""

from .standard_gbm import simulate_standard_gbm
from .brownian_bridge import simulate_brownian_bridge

__all__ = ['simulate_standard_gbm', 'simulate_brownian_bridge']
