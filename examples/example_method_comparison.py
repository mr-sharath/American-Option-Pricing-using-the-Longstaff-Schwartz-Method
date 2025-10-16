"""
Example 2: Comparing Standard GBM vs Brownian Bridge

This example compares the two simulation methods and validates
against Gustafsson benchmarks.
"""

import sys
sys.path.append('../src')

from american_option_pricer import AmericanOptionPricer
from benchmarks import compare_methods


def main():
    print("=" * 70)
    print("EXAMPLE 2: METHOD COMPARISON")
    print("=" * 70)
    
    # Create pricer with Gustafsson parameters
    pricer = AmericanOptionPricer(
        S0=100,
        K=100,
        r=0.03,
        sigma=0.15,
        T=1.0,
        N_steps=100,
        N_paths=100000
    )
    
    print("\nPricing American Put Option...")
    print(f"Parameters: {pricer.get_parameters()}")
    
    # Compare methods
    print("\n" + "-" * 70)
    comparison = pricer.compare_methods(option_type='put', seed=42)
    
    print(f"Standard GBM Price:     ${comparison['standard_gbm']:.6f}")
    print(f"Brownian Bridge Price:  ${comparison['brownian_bridge']:.6f}")
    print(f"Absolute Difference:    ${comparison['difference']:.6f}")
    print(f"Relative Difference:    {comparison['relative_diff_pct']:.3f}%")
    
    # Validate against benchmark
    print("\n" + "-" * 70)
    print("Validation Against Gustafsson Benchmark:")
    print("-" * 70)
    
    result = pricer.validate(
        option_type='put',
        method='brownian_bridge',
        benchmark_name='standard_case',
        seed=42
    )
    
    print(f"Benchmark Value:    ${result['benchmark_value']:.6f}")
    print(f"Calculated Value:   ${result['calculated_value']:.6f}")
    print(f"Absolute Error:     ${result['absolute_error']:.6f}")
    print(f"Relative Error:     {result['relative_error']:.3f}%")
    print(f"Status:             {result['status']}")
    
    if result['status'] == 'PASS':
        print("\n✅ Validation PASSED!")
    else:
        print("\n❌ Validation FAILED!")
    
    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
