"""
Benchmark validation tests.

Validates implementation against Gustafsson (2015) benchmark values.
"""

import sys
sys.path.append('../src')

import numpy as np
from simulation import simulate_standard_gbm, simulate_brownian_bridge
from pricing import price_american_option
from benchmarks import (
    GUSTAFSSON_BENCHMARKS,
    validate_against_benchmark,
    print_validation_report,
    get_benchmark_parameters,
    compare_methods
)


def test_standard_gbm_benchmark():
    """Test standard GBM against Gustafsson benchmark."""
    print("\n" + "=" * 70)
    print("TEST: Standard GBM vs Gustafsson Benchmark")
    print("=" * 70)
    
    params = get_benchmark_parameters('standard_case')
    
    np.random.seed(42)
    paths = simulate_standard_gbm(
        S0=params['S0'],
        r=params['r'],
        sigma=params['sigma'],
        T=params['T'],
        N_steps=params['N_steps'],
        N_paths=100000,
        seed=42
    )
    
    price = price_american_option(
        paths,
        K=params['K'],
        r=params['r'],
        T=params['T'],
        basis_degree=params['basis_degree'],
        option_type='put'
    )
    
    result = validate_against_benchmark(price, 'standard_case', 'american_put', tolerance=0.02)
    print_validation_report(result)
    
    assert result['status'] == 'PASS', f"Validation failed: {result['message']}"
    print("✅ test_standard_gbm_benchmark PASSED\n")


def test_brownian_bridge_benchmark():
    """Test Brownian Bridge against Gustafsson benchmark."""
    print("\n" + "=" * 70)
    print("TEST: Brownian Bridge vs Gustafsson Benchmark")
    print("=" * 70)
    
    params = get_benchmark_parameters('standard_case')
    
    np.random.seed(42)
    paths = simulate_brownian_bridge(
        S0=params['S0'],
        r=params['r'],
        sigma=params['sigma'],
        T=params['T'],
        N_steps=params['N_steps'],
        N_paths=100000,
        seed=42
    )
    
    price = price_american_option(
        paths,
        K=params['K'],
        r=params['r'],
        T=params['T'],
        basis_degree=params['basis_degree'],
        option_type='put'
    )
    
    result = validate_against_benchmark(price, 'standard_case', 'american_put', tolerance=0.02)
    print_validation_report(result)
    
    assert result['status'] == 'PASS', f"Validation failed: {result['message']}"
    print("✅ test_brownian_bridge_benchmark PASSED\n")


def test_method_comparison():
    """Compare standard GBM vs Brownian Bridge."""
    print("\n" + "=" * 70)
    print("TEST: Method Comparison")
    print("=" * 70)
    
    params = get_benchmark_parameters('standard_case')
    
    # Standard GBM
    np.random.seed(42)
    paths_std = simulate_standard_gbm(
        params['S0'], params['r'], params['sigma'], params['T'],
        params['N_steps'], 100000, seed=42
    )
    price_std = price_american_option(
        paths_std, params['K'], params['r'], params['T'],
        params['basis_degree'], 'put'
    )
    
    # Brownian Bridge
    np.random.seed(42)
    paths_bridge = simulate_brownian_bridge(
        params['S0'], params['r'], params['sigma'], params['T'],
        params['N_steps'], 100000, seed=42
    )
    price_bridge = price_american_option(
        paths_bridge, params['K'], params['r'], params['T'],
        params['basis_degree'], 'put'
    )
    
    compare_methods(price_std, price_bridge, 'standard_case')
    
    print("✅ test_method_comparison PASSED\n")


def test_itm_put():
    """Test in-the-money put pricing."""
    print("\n" + "=" * 70)
    print("TEST: In-the-Money Put (S0=90, K=100)")
    print("=" * 70)
    
    params = get_benchmark_parameters('itm_put')
    
    paths = simulate_brownian_bridge(
        params['S0'], params['r'], params['sigma'], params['T'],
        params['N_steps'], 100000, seed=42
    )
    
    price = price_american_option(
        paths, params['K'], params['r'], params['T'],
        params['basis_degree'], 'put'
    )
    
    print(f"ITM Put Price (S0=90): ${price:.4f}")
    
    # ITM put should be more expensive than ATM put
    assert price > 4.0, "ITM put should be worth more than $4"
    print("✅ test_itm_put PASSED\n")


def test_otm_put():
    """Test out-of-the-money put pricing."""
    print("\n" + "=" * 70)
    print("TEST: Out-of-the-Money Put (S0=110, K=100)")
    print("=" * 70)
    
    params = get_benchmark_parameters('otm_put')
    
    paths = simulate_brownian_bridge(
        params['S0'], params['r'], params['sigma'], params['T'],
        params['N_steps'], 100000, seed=42
    )
    
    price = price_american_option(
        paths, params['K'], params['r'], params['T'],
        params['basis_degree'], 'put'
    )
    
    print(f"OTM Put Price (S0=110): ${price:.4f}")
    
    # OTM put should be less expensive than ATM put
    assert price < 4.0, "OTM put should be worth less than $4"
    print("✅ test_otm_put PASSED\n")


def run_all_tests():
    """Run all benchmark tests."""
    print("\n" + "=" * 70)
    print("RUNNING BENCHMARK VALIDATION TESTS")
    print("=" * 70)
    
    test_standard_gbm_benchmark()
    test_brownian_bridge_benchmark()
    test_method_comparison()
    test_itm_put()
    test_otm_put()
    
    print("\n" + "=" * 70)
    print("✅ ALL BENCHMARK TESTS PASSED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_tests()
