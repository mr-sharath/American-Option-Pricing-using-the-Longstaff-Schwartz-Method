#!/usr/bin/env python3
"""
Test script to compare standard GBM simulation vs Brownian Bridge method.
"""

import sys
sys.path.append('src')

import numpy as np
import lsm_pricer
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters from Gustafsson paper
S0 = 100.0
K = 100.0
r = 0.03
sigma = 0.15
T = 1.0
N_steps = 100
N_paths = 100000
basis_degree = 3

print("=" * 70)
print("Testing Standard GBM vs Brownian Bridge Simulation")
print("=" * 70)
print(f"\nParameters:")
print(f"  S0 = {S0}, K = {K}, r = {r}, sigma = {sigma}, T = {T}")
print(f"  N_steps = {N_steps}, N_paths = {N_paths}, basis_degree = {basis_degree}")
print()

# Test 1: Standard GBM
print("1. Standard GBM Simulation...")
np.random.seed(42)
paths_standard = lsm_pricer.simulate_gbm_paths(S0, r, sigma, T, N_steps, N_paths)
price_standard = lsm_pricer.price_american_option(
    paths_standard, K, r, T, basis_degree, option_type='put'
)
print(f"   American Put Price (Standard): ${price_standard:.6f}")

# Test 2: Brownian Bridge
print("\n2. Brownian Bridge Simulation...")
np.random.seed(42)
paths_bridge = lsm_pricer.simulate_gbm_paths_brownian_bridge(S0, r, sigma, T, N_steps, N_paths)
price_bridge = lsm_pricer.price_american_option(
    paths_bridge, K, r, T, basis_degree, option_type='put'
)
print(f"   American Put Price (Bridge):   ${price_bridge:.6f}")

# Test 3: Using wrapper function
print("\n3. Using simulate_paths() wrapper...")
np.random.seed(42)
paths_wrapper_std = lsm_pricer.simulate_paths(S0, r, sigma, T, N_steps, N_paths, method='standard')
price_wrapper_std = lsm_pricer.price_american_option(
    paths_wrapper_std, K, r, T, basis_degree, option_type='put'
)
print(f"   Standard (via wrapper):        ${price_wrapper_std:.6f}")

np.random.seed(42)
paths_wrapper_bridge = lsm_pricer.simulate_paths(S0, r, sigma, T, N_steps, N_paths, method='brownian_bridge')
price_wrapper_bridge = lsm_pricer.price_american_option(
    paths_wrapper_bridge, K, r, T, basis_degree, option_type='put'
)
print(f"   Bridge (via wrapper):          ${price_wrapper_bridge:.6f}")

# Benchmark comparison
gustafsson_benchmark = 4.820608
print("\n" + "=" * 70)
print("Comparison with Gustafsson Benchmark")
print("=" * 70)
print(f"Gustafsson Benchmark:     ${gustafsson_benchmark:.6f}")
print(f"Standard GBM:             ${price_standard:.6f}  (error: {abs(price_standard - gustafsson_benchmark):.6f})")
print(f"Brownian Bridge:          ${price_bridge:.6f}  (error: {abs(price_bridge - gustafsson_benchmark):.6f})")

# Visualize sample paths
print("\n4. Generating comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot standard paths
np.random.seed(123)
paths_std_viz = lsm_pricer.simulate_gbm_paths(S0, r, sigma, T, N_steps, 50)
time_grid = np.linspace(0, T, N_steps + 1)
for i in range(50):
    axes[0].plot(time_grid, paths_std_viz[:, i], alpha=0.3, linewidth=0.8)
axes[0].axhline(y=K, color='r', linestyle='--', label='Strike Price', linewidth=2)
axes[0].set_xlabel('Time (years)')
axes[0].set_ylabel('Stock Price')
axes[0].set_title('Standard GBM Simulation (50 paths)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot Brownian Bridge paths
np.random.seed(123)
paths_bridge_viz = lsm_pricer.simulate_gbm_paths_brownian_bridge(S0, r, sigma, T, N_steps, 50)
for i in range(50):
    axes[1].plot(time_grid, paths_bridge_viz[:, i], alpha=0.3, linewidth=0.8)
axes[1].axhline(y=K, color='r', linestyle='--', label='Strike Price', linewidth=2)
axes[1].set_xlabel('Time (years)')
axes[1].set_ylabel('Stock Price')
axes[1].set_title('Brownian Bridge Simulation (50 paths)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/brownian_bridge_comparison.png', dpi=150, bbox_inches='tight')
print(f"   Plot saved to: results/brownian_bridge_comparison.png")

print("\n" + "=" * 70)
print("✅ All tests completed successfully!")
print("=" * 70)
