"""
Brownian Bridge simulation for Geometric Brownian Motion.

This module implements the Brownian Bridge variance reduction technique
for simulating stock price paths, as discussed in Gustafsson (2015).

The Brownian Bridge method:
1. First generates the terminal stock price S(T)
2. Recursively fills intermediate time points using conditional distributions
3. Provides variance reduction compared to standard sequential simulation

Reference:
    Gustafsson, W. (2015). "Evaluating the Longstaff-Schwartz method for 
    pricing of American options." Uppsala University.
    
    Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering."
    Springer, Section 2.5.2.
"""

import numpy as np


def simulate_brownian_bridge(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N_steps: int,
    N_paths: int,
    seed: int = None
) -> np.ndarray:
    """
    Simulates stock price paths using Brownian Bridge method.
    
    The Brownian Bridge is a variance reduction technique that:
    - Generates terminal value S(T) first
    - Fills intermediate points using conditional distributions
    - Reduces variance compared to sequential simulation
    
    For GBM, we work in log-space where the process is a Brownian motion
    with drift. The conditional distribution of log(S_t2) given log(S_t1)
    and log(S_t3) is Gaussian with:
    
    Mean: log(S_t1) + (r - 0.5*sigma^2)*(t2-t1) + 
          (t2-t1)/(t3-t1) * [log(S_t3) - log(S_t1) - (r - 0.5*sigma^2)*(t3-t1)]
    
    Variance: sigma^2 * (t2-t1) * (t3-t2) / (t3-t1)
    
    Args:
        S0 (float): Initial stock price at t=0.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the stock (annualized).
        T (float): Time to maturity in years.
        N_steps (int): Number of time steps for discretization.
        N_paths (int): Number of Monte Carlo paths to simulate.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Array of shape (N_steps + 1, N_paths) containing
                    simulated stock price paths using Brownian Bridge.
    
    Example:
        >>> paths = simulate_brownian_bridge(
        ...     S0=100, r=0.03, sigma=0.15, T=1.0,
        ...     N_steps=100, N_paths=10000, seed=42
        ... )
        >>> paths.shape
        (101, 10000)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N_steps
    
    # Initialize paths array
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0
    
    # Step 1: Generate terminal values S(T) directly
    Z_T = np.random.standard_normal(N_paths)
    paths[-1] = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_T)
    
    # Step 2: Recursively fill intermediate points using bridge construction
    _fill_bridge_recursive(paths, 0, N_steps, N_paths, r, sigma, dt)
    
    return paths


def _fill_bridge_recursive(
    paths: np.ndarray,
    left_idx: int,
    right_idx: int,
    N_paths: int,
    r: float,
    sigma: float,
    dt: float
) -> None:
    """
    Recursively fills the Brownian Bridge between two time points.
    
    This is an internal helper function that implements the recursive
    bisection algorithm for the Brownian Bridge construction.
    
    Args:
        paths (np.ndarray): The paths array to fill (modified in-place).
        left_idx (int): Left time index (already filled).
        right_idx (int): Right time index (already filled).
        N_paths (int): Number of paths.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        dt (float): Time step size.
    """
    # Base case: adjacent time points, nothing to fill
    if right_idx - left_idx <= 1:
        return
    
    # Find the midpoint index
    mid_idx = (left_idx + right_idx) // 2
    
    # Calculate time values
    t_left = left_idx * dt
    t_mid = mid_idx * dt
    t_right = right_idx * dt
    
    # Time differences
    dt_left_mid = t_mid - t_left
    dt_mid_right = t_right - t_mid
    dt_left_right = t_right - t_left
    
    # Get stock prices at left and right endpoints
    S_left = paths[left_idx]
    S_right = paths[right_idx]
    
    # Work in log-space for GBM (Brownian motion with drift)
    log_S_left = np.log(S_left)
    log_S_right = np.log(S_right)
    
    # Conditional mean for log(S_mid) given log(S_left) and log(S_right)
    # This is the key formula for the Brownian Bridge
    mu_cond = log_S_left + (r - 0.5 * sigma**2) * dt_left_mid + \
              (dt_left_mid / dt_left_right) * \
              (log_S_right - log_S_left - (r - 0.5 * sigma**2) * dt_left_right)
    
    # Conditional standard deviation
    sigma_cond = sigma * np.sqrt(dt_left_mid * dt_mid_right / dt_left_right)
    
    # Generate random increments
    Z = np.random.standard_normal(N_paths)
    
    # Simulate the midpoint in log-space
    log_S_mid = mu_cond + sigma_cond * Z
    
    # Convert back to price space
    paths[mid_idx] = np.exp(log_S_mid)
    
    # Recursively fill left and right sub-intervals
    _fill_bridge_recursive(paths, left_idx, mid_idx, N_paths, r, sigma, dt)
    _fill_bridge_recursive(paths, mid_idx, right_idx, N_paths, r, sigma, dt)


def get_bridge_order(N_steps: int) -> list:
    """
    Returns the order in which time points are filled in the bridge construction.
    
    This is useful for understanding or visualizing the bridge algorithm.
    
    Args:
        N_steps (int): Number of time steps.
    
    Returns:
        list: List of indices in the order they are filled.
    
    Example:
        >>> get_bridge_order(8)
        [4, 2, 6, 1, 3, 5, 7]
    """
    filled = []
    
    def _get_order(left, right):
        if right - left <= 1:
            return
        mid = (left + right) // 2
        filled.append(mid)
        _get_order(left, mid)
        _get_order(mid, right)
    
    _get_order(0, N_steps)
    return filled
