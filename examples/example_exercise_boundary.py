"""
Example 3: Computing Exercise Boundary

This example demonstrates how to compute and visualize the optimal
exercise boundary for American options.
"""

import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from american_option_pricer import AmericanOptionPricer


def main():
    print("=" * 70)
    print("EXAMPLE 3: EXERCISE BOUNDARY COMPUTATION")
    print("=" * 70)
    
    # Create pricer
    pricer = AmericanOptionPricer(
        S0=100,
        K=100,
        r=0.03,
        sigma=0.15,
        T=1.0,
        N_steps=100,
        N_paths=100000
    )
    
    print("\nComputing exercise boundary for American Put...")
    
    # Compute price and boundary
    price, boundary = pricer.price_with_exercise_boundary(
        option_type='put',
        method='brownian_bridge',
        seed=42
    )
    
    print(f"American Put Price: ${price:.4f}")
    print(f"Boundary computed at {np.sum(~np.isnan(boundary))} time points")
    
    # Plot the boundary
    print("\nGenerating plot...")
    
    time_grid = np.linspace(0, pricer.T, pricer.N_steps + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, boundary, 'b-', linewidth=2, label='Exercise Boundary')
    plt.axhline(y=pricer.K, color='r', linestyle='--', linewidth=1.5, label=f'Strike Price (K={pricer.K})')
    plt.xlabel('Time to Maturity (years)', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.title('American Put Option - Optimal Exercise Boundary', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_file = '../results/exercise_boundary_example.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Print some boundary values
    print("\nSample Boundary Values:")
    print("-" * 40)
    for t_idx in [25, 50, 75, 99]:
        if not np.isnan(boundary[t_idx]):
            print(f"  t = {time_grid[t_idx]:.2f} years: S* = ${boundary[t_idx]:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
