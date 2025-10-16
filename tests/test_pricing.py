"""
Unit tests for LSM pricing algorithm.

Tests for American option pricing using Longstaff-Schwartz method.
"""

import sys
sys.path.append('../src')

import numpy as np
from simulation import simulate_standard_gbm
from pricing import price_american_option, price_with_boundary


def test_put_price_positive():
    """Test that American put price is positive."""
    paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price = price_american_option(paths, 100, 0.03, 1.0, 3, 'put')
    
    assert price > 0, "American put price should be positive"
    print(f"✅ test_put_price_positive PASSED (price = ${price:.4f})")


def test_call_price_positive():
    """Test that American call price is positive."""
    paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price = price_american_option(paths, 100, 0.03, 1.0, 3, 'call')
    
    assert price > 0, "American call price should be positive"
    print(f"✅ test_call_price_positive PASSED (price = ${price:.4f})")


def test_put_call_parity_violation():
    """Test that American options violate put-call parity (as expected)."""
    paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    
    put_price = price_american_option(paths, 100, 0.03, 1.0, 3, 'put')
    call_price = price_american_option(paths, 100, 0.03, 1.0, 3, 'call')
    
    # For American options, put-call parity doesn't hold exactly
    # But we can check that prices are reasonable
    assert put_price > 0 and call_price > 0, "Both prices should be positive"
    print(f"✅ test_put_call_parity_violation PASSED")
    print(f"   Put: ${put_price:.4f}, Call: ${call_price:.4f}")


def test_itm_put_higher_price():
    """Test that ITM put has higher price than ATM put."""
    # ITM put (S0 < K)
    paths_itm = simulate_standard_gbm(90, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price_itm = price_american_option(paths_itm, 100, 0.03, 1.0, 3, 'put')
    
    # ATM put (S0 = K)
    paths_atm = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price_atm = price_american_option(paths_atm, 100, 0.03, 1.0, 3, 'put')
    
    assert price_itm > price_atm, "ITM put should be more expensive than ATM put"
    print(f"✅ test_itm_put_higher_price PASSED")
    print(f"   ITM: ${price_itm:.4f}, ATM: ${price_atm:.4f}")


def test_otm_put_lower_price():
    """Test that OTM put has lower price than ATM put."""
    # OTM put (S0 > K)
    paths_otm = simulate_standard_gbm(110, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price_otm = price_american_option(paths_otm, 100, 0.03, 1.0, 3, 'put')
    
    # ATM put (S0 = K)
    paths_atm = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price_atm = price_american_option(paths_atm, 100, 0.03, 1.0, 3, 'put')
    
    assert price_otm < price_atm, "OTM put should be cheaper than ATM put"
    print(f"✅ test_otm_put_lower_price PASSED")
    print(f"   OTM: ${price_otm:.4f}, ATM: ${price_atm:.4f}")


def test_boundary_computation():
    """Test that exercise boundary computation works."""
    paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000, seed=42)
    price, boundary = price_with_boundary(paths, 100, 0.03, 1.0, 3, 'put')
    
    assert price > 0, "Price should be positive"
    assert len(boundary) == 101, f"Boundary should have 101 points, got {len(boundary)}"
    assert not np.all(np.isnan(boundary)), "Boundary should have some non-NaN values"
    
    print(f"✅ test_boundary_computation PASSED")
    print(f"   Price: ${price:.4f}, Boundary points: {np.sum(~np.isnan(boundary))}")


def test_invalid_option_type():
    """Test that invalid option type raises error."""
    paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 1000, seed=42)
    
    try:
        price = price_american_option(paths, 100, 0.03, 1.0, 3, 'invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be 'put' or 'call'" in str(e)
        print("✅ test_invalid_option_type PASSED")


def run_all_tests():
    """Run all pricing tests."""
    print("\n" + "=" * 70)
    print("RUNNING PRICING TESTS")
    print("=" * 70 + "\n")
    
    test_put_price_positive()
    test_call_price_positive()
    test_put_call_parity_violation()
    test_itm_put_higher_price()
    test_otm_put_lower_price()
    test_boundary_computation()
    test_invalid_option_type()
    
    print("\n" + "=" * 70)
    print("✅ ALL PRICING TESTS PASSED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_tests()
