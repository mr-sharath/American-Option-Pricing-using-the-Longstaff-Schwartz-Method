"""
Pricing module for American options using Longstaff-Schwartz Method.

This module implements the LSM algorithm for pricing American-style options
using Monte Carlo simulation and least-squares regression.
"""

from .lsm_algorithm import price_american_option, price_with_boundary

__all__ = ['price_american_option', 'price_with_boundary']
