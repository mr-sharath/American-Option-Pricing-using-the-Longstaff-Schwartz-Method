"""
Longstaff-Schwartz Method (LSM) for American Option Pricing.

This module implements the LSM algorithm as described in:
- Longstaff & Schwartz (2001)
- Gustafsson (2015)

The algorithm uses backward induction with least-squares regression to
estimate continuation values and determine optimal exercise decisions.

Reference:
    Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American Options 
    by Simulation: A Simple Least-Squares Approach." The Review of Financial 
    Studies, 14(1), 113-147.
    
    Gustafsson, W. (2015). "Evaluating the Longstaff-Schwartz method for 
    pricing of American options." Uppsala University.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from basis_functions import laguerre_polynomials


def price_american_option(
    stock_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    basis_degree: int,
    option_type: str = 'put'
) -> float:
    """
    Prices an American option using the Longstaff-Schwartz Method (LSM).
    
    The LSM algorithm:
    1. Initialize payoffs at maturity T
    2. Backward induction from T-1 to t=1:
       a. Discount future cashflows
       b. For in-the-money paths, regress continuation values on basis functions
       c. Compare immediate exercise vs. continuation
       d. Update cashflows for paths where early exercise is optimal
    3. Discount to t=0 and average to get option price
    
    Args:
        stock_paths (np.ndarray): Simulated stock price paths of shape 
                                  (N_steps + 1, N_paths).
        K (float): Strike price of the option.
        r (float): Risk-free interest rate (annualized).
        T (float): Time to maturity in years.
        basis_degree (int): Degree of Laguerre polynomials for regression.
                           Gustafsson uses degree 3.
        option_type (str): Type of option, either 'put' or 'call'.
    
    Returns:
        float: The estimated price of the American option.
    
    Raises:
        ValueError: If option_type is not 'put' or 'call'.
    
    Example:
        >>> import numpy as np
        >>> from simulation import simulate_standard_gbm
        >>> paths = simulate_standard_gbm(100, 0.03, 0.15, 1.0, 100, 10000)
        >>> price = price_american_option(paths, 100, 0.03, 1.0, 3, 'put')
        >>> print(f"American Put Price: ${price:.4f}")
    """
    N_steps, N_paths = stock_paths.shape[0] - 1, stock_paths.shape[1]
    dt = T / N_steps
    
    # Step 1: Initialize payoff at maturity (time T)
    if option_type == 'put':
        payoff = np.maximum(K - stock_paths[-1], 0)
    elif option_type == 'call':
        payoff = np.maximum(stock_paths[-1] - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    # Cashflow array tracks the value received on each path
    cashflow = payoff.copy()
    
    # Step 2: Backward Induction (from t = N-1 down to t = 1)
    for t in range(N_steps - 1, 0, -1):
        # Discount future cashflows back one time step
        cashflow = cashflow * np.exp(-r * dt)
        
        # Current stock prices at time t
        S_t = stock_paths[t]
        
        # Identify in-the-money (ITM) paths
        if option_type == 'put':
            itm_mask = (K - S_t) > 0
        else:  # call
            itm_mask = (S_t - K) > 0
        
        itm_paths_idx = np.where(itm_mask)[0]
        
        # Only perform regression if there are ITM paths
        if len(itm_paths_idx) > 0:
            # Extract ITM stock prices and corresponding cashflows
            X = S_t[itm_paths_idx]
            Y = cashflow[itm_paths_idx]
            
            # Generate basis functions (Laguerre polynomials)
            L = laguerre_polynomials(X, basis_degree)
            
            # Perform least-squares regression: beta = (L^T L)^-1 L^T Y
            # This estimates the continuation value function
            beta = np.linalg.lstsq(L, Y, rcond=None)[0]
            
            # Estimate continuation values for ITM paths
            continuation_value = L @ beta
            
            # Calculate immediate exercise value
            if option_type == 'put':
                exercise_value = K - X
            else:  # call
                exercise_value = X - K
            
            # Determine where immediate exercise is optimal
            exercise_mask = exercise_value > continuation_value
            exercise_idx = itm_paths_idx[exercise_mask]
            
            # Update cashflows: replace with exercise value if optimal
            cashflow[exercise_idx] = exercise_value[exercise_mask]
    
    # Step 3: Discount cashflows to t=0 and compute option price
    option_price = np.mean(cashflow * np.exp(-r * dt))
    
    return option_price


def price_with_boundary(
    stock_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    basis_degree: int,
    option_type: str = 'put'
) -> tuple:
    """
    Prices an American option and computes the exercise boundary.
    
    The exercise boundary is the critical stock price at each time step
    below which (for puts) or above which (for calls) early exercise is optimal.
    
    Args:
        stock_paths (np.ndarray): Simulated stock price paths.
        K (float): Strike price.
        r (float): Risk-free rate.
        T (float): Time to maturity.
        basis_degree (int): Degree of Laguerre polynomials.
        option_type (str): 'put' or 'call'.
    
    Returns:
        tuple: (option_price, exercise_boundary)
            - option_price (float): Estimated option price
            - exercise_boundary (np.ndarray): Array of length N_steps containing
                                             the exercise boundary at each time step
    
    Example:
        >>> price, boundary = price_with_boundary(paths, 100, 0.03, 1.0, 3, 'put')
        >>> print(f"Price: ${price:.4f}")
        >>> print(f"Boundary at t=0.5: ${boundary[50]:.2f}")
    """
    N_steps, N_paths = stock_paths.shape[0] - 1, stock_paths.shape[1]
    dt = T / N_steps
    
    # Create a stock price grid for boundary calculation
    S_min = min(K * 0.5, np.min(stock_paths))
    S_max = max(K * 1.5, np.max(stock_paths))
    S_grid = np.linspace(S_min, S_max, 200)
    
    # Initialize boundary array (NaN for time points without regression)
    exercise_boundary = np.full(N_steps + 1, np.nan)
    
    # Initialize payoff at maturity
    if option_type == 'put':
        payoff = np.maximum(K - stock_paths[-1], 0)
    else:
        payoff = np.maximum(stock_paths[-1] - K, 0)
    
    cashflow = payoff.copy()
    
    # Backward induction
    for t in range(N_steps - 1, 0, -1):
        cashflow = cashflow * np.exp(-r * dt)
        S_t = stock_paths[t]
        
        # Find ITM paths
        if option_type == 'put':
            itm_mask = (K - S_t) > 0
        else:
            itm_mask = (S_t - K) > 0
        
        itm_paths_idx = np.where(itm_mask)[0]
        
        if len(itm_paths_idx) > 0:
            X = S_t[itm_paths_idx]
            Y = cashflow[itm_paths_idx]
            
            # Regression
            L = laguerre_polynomials(X, basis_degree)
            beta = np.linalg.lstsq(L, Y, rcond=None)[0]
            
            # --- Boundary Calculation ---
            # Evaluate continuation value on grid
            L_grid = laguerre_polynomials(S_grid, basis_degree)
            continuation_value_grid = L_grid @ beta
            
            if option_type == 'put':
                exercise_value_grid = np.maximum(K - S_grid, 0)
                # Find lowest S where exercise is optimal
                exercise_optimal = exercise_value_grid > continuation_value_grid
                if np.any(exercise_optimal):
                    exercise_boundary[t] = S_grid[np.where(exercise_optimal)[0][0]]
                else:
                    exercise_boundary[t] = 0  # Never optimal
            else:  # call
                exercise_value_grid = np.maximum(S_grid - K, 0)
                # Find highest S where exercise is optimal
                exercise_optimal = exercise_value_grid > continuation_value_grid
                if np.any(exercise_optimal):
                    exercise_boundary[t] = S_grid[np.where(exercise_optimal)[0][-1]]
                else:
                    exercise_boundary[t] = np.inf  # Never optimal
            
            # --- Continue with pricing ---
            continuation_value = L @ beta
            
            if option_type == 'put':
                exercise_value = K - X
            else:
                exercise_value = X - K
            
            exercise_mask = exercise_value > continuation_value
            exercise_idx = itm_paths_idx[exercise_mask]
            cashflow[exercise_idx] = exercise_value[exercise_mask]
    
    # Final discounting
    option_price = np.mean(cashflow * np.exp(-r * dt))
    
    return option_price, exercise_boundary
