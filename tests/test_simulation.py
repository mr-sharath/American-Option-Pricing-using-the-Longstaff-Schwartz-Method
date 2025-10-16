"""
Unit tests for simulation modules.

Tests for standard GBM and Brownian Bridge simulation methods.
"""

import sys
sys.path.append('../src')

import numpy as np
from simulation import simulate_standard_gbm, simulate_brownian_bridge


def test_standard_gbm_shape():
    """Test that standard GBM produces correct output shape."""
    S0, r, sigma, T = 100, 0.03, 0.15, 1.0
    N_steps, N_paths = 100, 1000
    
    paths = simulate_standard_gbm(S0, r, sigma, T, N_steps, N_paths, seed=42)
    
    assert paths.shape == (N_steps + 1, N_paths), \
        f"Expected shape ({N_steps + 1}, {N_paths}), got {paths.shape}"
    print("✅ test_standard_gbm_shape PASSED")


def test_standard_gbm_initial_value():
    """Test that all paths start at S0."""
    S0 = 100
    paths = simulate_standard_gbm(S0, 0.03, 0.15, 1.0, 100, 1000, seed=42)
    
    assert np.all(paths[0] == S0), "All paths should start at S0"
    print("✅ test_standard_gbm_initial_value PASSED")


def test_standard_gbm_positive():
    """Test that GBM paths remain positive."""
    paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 1000, seed=42)
    
    assert np.all(paths > 0), "All stock prices should be positive"
    print("✅ test_standard_gbm_positive PASSED")


def test_brownian_bridge_shape():
    """Test that Brownian Bridge produces correct output shape."""
    S0, r, sigma, T = 100, 0.03, 0.15, 1.0
    N_steps, N_paths = 100, 1000
    
    paths = simulate_brownian_bridge(S0, r, sigma, T, N_steps, N_paths, seed=42)
    
    assert paths.shape == (N_steps + 1, N_paths), \
        f"Expected shape ({N_steps + 1}, {N_paths}), got {paths.shape}"
    print("✅ test_brownian_bridge_shape PASSED")


def test_brownian_bridge_initial_value():
    """Test that all paths start at S0."""
    S0 = 100
    paths = simulate_brownian_bridge(S0, 0.03, 0.15, 1.0, 100, 1000, seed=42)
    
    assert np.all(paths[0] == S0), "All paths should start at S0"
    print("✅ test_brownian_bridge_initial_value PASSED")


def test_brownian_bridge_positive():
    """Test that Brownian Bridge paths remain positive."""
    paths = simulate_brownian_bridge(100, 0.03, 0.15, 1.0, 100, 1000, seed=42)
    
    assert np.all(paths > 0), "All stock prices should be positive"
    print("✅ test_brownian_bridge_positive PASSED")


def test_reproducibility():
    """Test that setting seed produces reproducible results."""
    params = (100, 0.03, 0.15, 1.0, 100, 1000)
    
    paths1 = simulate_standard_gbm(*params, seed=42)
    paths2 = simulate_standard_gbm(*params, seed=42)
    
    assert np.allclose(paths1, paths2), "Same seed should produce identical results"
    print("✅ test_reproducibility PASSED")


def test_terminal_distribution():
    """Test that terminal values follow lognormal distribution approximately."""
    S0, r, sigma, T = 100, 0.05, 0.2, 1.0
    N_paths = 50000
    
    paths = simulate_standard_gbm(S0, r, sigma, T, 100, N_paths, seed=42)
    terminal_values = paths[-1]
    
    # Theoretical mean and variance of S(T)
    expected_mean = S0 * np.exp(r * T)
    expected_var = S0**2 * np.exp(2*r*T) * (np.exp(sigma**2 * T) - 1)
    
    sample_mean = np.mean(terminal_values)
    sample_var = np.var(terminal_values)
    
    # Allow 5% tolerance
    assert abs(sample_mean - expected_mean) / expected_mean < 0.05, \
        f"Mean mismatch: expected {expected_mean:.2f}, got {sample_mean:.2f}"
    assert abs(sample_var - expected_var) / expected_var < 0.10, \
        f"Variance mismatch: expected {expected_var:.2f}, got {sample_var:.2f}"
    
    print("✅ test_terminal_distribution PASSED")


def run_all_tests():
    """Run all simulation tests."""
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION TESTS")
    print("=" * 70 + "\n")
    
    test_standard_gbm_shape()
    test_standard_gbm_initial_value()
    test_standard_gbm_positive()
    test_brownian_bridge_shape()
    test_brownian_bridge_initial_value()
    test_brownian_bridge_positive()
    test_reproducibility()
    test_terminal_distribution()
    
    print("\n" + "=" * 70)
    print("✅ ALL SIMULATION TESTS PASSED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_tests()
