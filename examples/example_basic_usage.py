"""
Example 1: Basic Usage

This example demonstrates the simplest way to price American options
using the modular interface.
"""

import sys
sys.path.append('../src')

from american_option_pricer import AmericanOptionPricer, quick_price


def main():
    print("=" * 70)
    print("EXAMPLE 1: BASIC USAGE")
    print("=" * 70)
    
    # Method 1: Quick one-liner
    print("\n1. Quick Price (one-liner):")
    price = quick_price(S0=100, K=100, r=0.03, sigma=0.15, T=1.0, option_type='put')
    print(f"   American Put Price: ${price:.4f}")
    
    # Method 2: Using the pricer class
    print("\n2. Using AmericanOptionPricer class:")
    pricer = AmericanOptionPricer(
        S0=100,      # Initial stock price
        K=100,       # Strike price
        r=0.03,      # Risk-free rate (3%)
        sigma=0.15,  # Volatility (15%)
        T=1.0,       # 1 year to maturity
        N_steps=100,
        N_paths=100000
    )
    
    # Price a put option
    put_price = pricer.price(option_type='put', method='brownian_bridge', seed=42)
    print(f"   American Put Price:  ${put_price:.4f}")
    
    # Price a call option
    call_price = pricer.price(option_type='call', method='brownian_bridge', seed=42)
    print(f"   American Call Price: ${call_price:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
