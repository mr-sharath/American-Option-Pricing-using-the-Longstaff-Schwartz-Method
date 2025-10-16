"""
Benchmark values from Gustafsson (2015) paper.

This module contains the benchmark American option prices reported in:
    Gustafsson, W. (2015). "Evaluating the Longstaff-Schwartz method for 
    pricing of American options." Uppsala University.

The benchmarks are from Table 3.1 (page 411 in citations).
"""

import numpy as np
from typing import Dict, Tuple


# Benchmark values from Gustafsson (2015), Table 3.1
GUSTAFSSON_BENCHMARKS = {
    'standard_case': {
        'description': 'Standard benchmark case from Gustafsson Table 3.1',
        'parameters': {
            'S0': 100.0,
            'K': 100.0,
            'r': 0.03,
            'sigma': 0.15,
            'T': 1.0,
            'N_steps': 100,
            'basis_degree': 3
        },
        'american_put': 4.820608,  # Gustafsson benchmark value
        'european_put': None,  # Not provided in paper
    },
    'itm_put': {
        'description': 'In-the-money put (S0 < K)',
        'parameters': {
            'S0': 90.0,
            'K': 100.0,
            'r': 0.03,
            'sigma': 0.15,
            'T': 1.0,
            'N_steps': 100,
            'basis_degree': 3
        },
        'american_put': None,  # Can be computed for validation
        'european_put': None,
    },
    'otm_put': {
        'description': 'Out-of-the-money put (S0 > K)',
        'parameters': {
            'S0': 110.0,
            'K': 100.0,
            'r': 0.03,
            'sigma': 0.15,
            'T': 1.0,
            'N_steps': 100,
            'basis_degree': 3
        },
        'american_put': None,
        'european_put': None,
    }
}


def validate_against_benchmark(
    calculated_price: float,
    benchmark_name: str = 'standard_case',
    option_type: str = 'american_put',
    tolerance: float = 0.01
) -> Dict[str, any]:
    """
    Validates a calculated option price against Gustafsson benchmarks.
    
    Args:
        calculated_price (float): The option price calculated by your implementation.
        benchmark_name (str): Name of the benchmark case (default: 'standard_case').
        option_type (str): Type of option (default: 'american_put').
        tolerance (float): Acceptable relative error tolerance (default: 0.01 = 1%).
    
    Returns:
        dict: Validation results containing:
            - 'benchmark_value': The benchmark price
            - 'calculated_value': Your calculated price
            - 'absolute_error': Absolute difference
            - 'relative_error': Relative error as percentage
            - 'within_tolerance': Boolean indicating if within tolerance
            - 'status': 'PASS' or 'FAIL'
    
    Example:
        >>> result = validate_against_benchmark(4.8140, 'standard_case', 'american_put')
        >>> print(f"Status: {result['status']}")
        >>> print(f"Error: {result['relative_error']:.2f}%")
    """
    if benchmark_name not in GUSTAFSSON_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    benchmark = GUSTAFSSON_BENCHMARKS[benchmark_name]
    benchmark_value = benchmark.get(option_type)
    
    if benchmark_value is None:
        return {
            'benchmark_value': None,
            'calculated_value': calculated_price,
            'absolute_error': None,
            'relative_error': None,
            'within_tolerance': None,
            'status': 'NO_BENCHMARK',
            'message': f'No benchmark available for {option_type} in {benchmark_name}'
        }
    
    absolute_error = abs(calculated_price - benchmark_value)
    relative_error = (absolute_error / benchmark_value) * 100
    within_tolerance = relative_error <= (tolerance * 100)
    
    return {
        'benchmark_value': benchmark_value,
        'calculated_value': calculated_price,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'within_tolerance': within_tolerance,
        'status': 'PASS' if within_tolerance else 'FAIL',
        'message': f'Relative error: {relative_error:.3f}%'
    }


def print_validation_report(validation_result: Dict) -> None:
    """
    Prints a formatted validation report.
    
    Args:
        validation_result (dict): Result from validate_against_benchmark().
    """
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    
    if validation_result['status'] == 'NO_BENCHMARK':
        print(validation_result['message'])
        return
    
    print(f"Benchmark Value:    ${validation_result['benchmark_value']:.6f}")
    print(f"Calculated Value:   ${validation_result['calculated_value']:.6f}")
    print(f"Absolute Error:     ${validation_result['absolute_error']:.6f}")
    print(f"Relative Error:     {validation_result['relative_error']:.3f}%")
    print(f"Status:             {validation_result['status']}")
    
    if validation_result['status'] == 'PASS':
        print("\n✅ VALIDATION PASSED - Result within acceptable tolerance")
    else:
        print("\n❌ VALIDATION FAILED - Result outside acceptable tolerance")
    
    print("=" * 70)


def get_benchmark_parameters(benchmark_name: str = 'standard_case') -> Dict:
    """
    Retrieves the parameters for a specific benchmark case.
    
    Args:
        benchmark_name (str): Name of the benchmark case.
    
    Returns:
        dict: Dictionary of parameters (S0, K, r, sigma, T, etc.)
    
    Example:
        >>> params = get_benchmark_parameters('standard_case')
        >>> print(f"S0 = {params['S0']}, K = {params['K']}")
    """
    if benchmark_name not in GUSTAFSSON_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return GUSTAFSSON_BENCHMARKS[benchmark_name]['parameters'].copy()


def compare_methods(
    standard_price: float,
    bridge_price: float,
    benchmark_name: str = 'standard_case'
) -> None:
    """
    Compares standard GBM and Brownian Bridge results against benchmark.
    
    Args:
        standard_price (float): Price from standard GBM simulation.
        bridge_price (float): Price from Brownian Bridge simulation.
        benchmark_name (str): Name of benchmark case.
    """
    benchmark_value = GUSTAFSSON_BENCHMARKS[benchmark_name]['american_put']
    
    if benchmark_value is None:
        print("No benchmark available for comparison")
        return
    
    std_error = abs(standard_price - benchmark_value)
    bridge_error = abs(bridge_price - benchmark_value)
    improvement = ((std_error - bridge_error) / std_error) * 100
    
    print("=" * 70)
    print("METHOD COMPARISON")
    print("=" * 70)
    print(f"Benchmark Value:        ${benchmark_value:.6f}")
    print(f"\nStandard GBM:           ${standard_price:.6f}")
    print(f"  Error:                ${std_error:.6f} ({(std_error/benchmark_value)*100:.3f}%)")
    print(f"\nBrownian Bridge:        ${bridge_price:.6f}")
    print(f"  Error:                ${bridge_error:.6f} ({(bridge_error/benchmark_value)*100:.3f}%)")
    print(f"\nImprovement:            {improvement:.1f}%")
    
    if improvement > 0:
        print("\n✅ Brownian Bridge shows lower error")
    elif improvement < 0:
        print("\n⚠️  Standard GBM shows lower error (unusual)")
    else:
        print("\n➖ Both methods show equal error")
    
    print("=" * 70)
