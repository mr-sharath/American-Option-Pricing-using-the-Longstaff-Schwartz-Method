"""
Benchmark module for validation of American option pricing.

This module contains benchmark values from academic papers and
validation functions to compare implementation results.
"""

from .gustafsson_benchmarks import (
    GUSTAFSSON_BENCHMARKS,
    validate_against_benchmark,
    print_validation_report,
    get_benchmark_parameters,
    compare_methods
)

__all__ = [
    'GUSTAFSSON_BENCHMARKS',
    'validate_against_benchmark',
    'print_validation_report',
    'get_benchmark_parameters',
    'compare_methods'
]
