"""
Standard Geometric Brownian Motion (GBM) simulation.

This module implements the standard sequential forward simulation of stock
price paths under GBM, as described in Gustafsson (2015), equation 1.5.

Reference:
    Gustafsson, W. (2015). "Evaluating the Longstaff-Schwartz method for 
    pricing of American options." Uppsala University.
"""

import numpy as np


def simulate_standard_gbm(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N_steps: int,
    N_paths: int,
    seed: int = None
) -> np.ndarray:
    """
    Simulates stock price paths using standard Geometric Brownian Motion.
    
    The stock price follows the stochastic differential equation:
        dS_t = r * S_t * dt + sigma * S_t * dW_t
    
    Discretized using Euler scheme (Gustafsson eq. 1.5):
        S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    
    where Z ~ N(0,1) is a standard normal random variable.
    
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
                    simulated stock price paths. Each column represents
                    one path, each row represents a time point.
    
    Example:
        >>> paths = simulate_standard_gbm(
        ...     S0=100, r=0.03, sigma=0.15, T=1.0,
        ...     N_steps=100, N_paths=10000, seed=42
        ... )
        >>> paths.shape
        (101, 10000)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N_steps  # Time step size
    
    # Initialize paths array: rows = time points, columns = paths
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0  # Set initial price for all paths
    
    # Sequential forward simulation
    for t in range(1, N_steps + 1):
        # Generate standard normal random variables
        Z = np.random.standard_normal(N_paths)
        
        # Apply GBM formula (Gustafsson eq. 1.5)
        paths[t] = paths[t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    
    return paths


def get_time_grid(T: float, N_steps: int) -> np.ndarray:
    """
    Generate time grid for the simulation.
    
    Args:
        T (float): Time to maturity.
        N_steps (int): Number of time steps.
    
    Returns:
        np.ndarray: Array of time points from 0 to T.
    """
    return np.linspace(0, T, N_steps + 1)
