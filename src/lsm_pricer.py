# File: src/lsm_pricer.py

import numpy as np
from basis_functions import laguerre_polynomials

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N_steps: int,
    N_paths: int
) -> np.ndarray:
    """
    Simulates stock price paths using Geometric Brownian Motion.

    Args:
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.
        T (float): Time to maturity (in years).
        N_steps (int): Number of time steps for the simulation.
        N_paths (int): Number of simulation paths to generate.

    Returns:
        np.ndarray: A 2D numpy array of shape (N_steps + 1, N_paths)
                    containing the simulated stock price paths.
    """
    dt = T / N_steps  # Time step size

    # Initialize a numpy array to store the paths
    # The shape should be (N_steps + 1) rows for time points, and N_paths columns
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0  # Set the initial price for all paths

    # Loop from the first time step to the last
    for t in range(1, N_steps + 1):
        # Generate random standard normal variables for each path
        Z = np.random.standard_normal(N_paths)

        # Apply the GBM formula (Gustaffson, eq. 1.5)
        # S_t = S_{t-1} * exp(...)
        paths[t] = paths[t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return paths

def simulate_gbm_paths_brownian_bridge(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N_steps: int,
    N_paths: int
) -> np.ndarray:
    """
    Simulates stock price paths using Geometric Brownian Motion with Brownian Bridge.
    
    The Brownian Bridge method is a variance reduction technique that:
    1. First simulates the terminal stock price S(T)
    2. Then recursively fills in intermediate time points using conditional distributions
    
    This approach can reduce variance compared to sequential simulation.
    
    Args:
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.
        T (float): Time to maturity (in years).
        N_steps (int): Number of time steps for the simulation.
        N_paths (int): Number of simulation paths to generate.
    
    Returns:
        np.ndarray: A 2D numpy array of shape (N_steps + 1, N_paths)
                    containing the simulated stock price paths.
    """
    dt = T / N_steps
    
    # Initialize paths array
    paths = np.zeros((N_steps + 1, N_paths))
    paths[0] = S0
    
    # Step 1: Generate terminal values S(T) directly
    Z_T = np.random.standard_normal(N_paths)
    paths[-1] = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_T)
    
    # Step 2: Use Brownian Bridge to fill in intermediate points
    # We recursively bisect intervals and fill in midpoints
    def fill_bridge(paths, left_idx, right_idx, N_paths, r, sigma, dt):
        """
        Recursively fills in the Brownian Bridge between two time points.
        
        Args:
            paths: The paths array to fill
            left_idx: Left time index (already filled)
            right_idx: Right time index (already filled)
            N_paths: Number of paths
            r: Risk-free rate
            sigma: Volatility
            dt: Time step size
        """
        if right_idx - left_idx <= 1:
            return
        
        # Find the midpoint
        mid_idx = (left_idx + right_idx) // 2
        
        # Time intervals
        t_left = left_idx * dt
        t_mid = mid_idx * dt
        t_right = right_idx * dt
        
        # Time differences
        dt_left_mid = t_mid - t_left
        dt_mid_right = t_right - t_mid
        dt_left_right = t_right - t_left
        
        # Get the left and right values (in log space for GBM)
        S_left = paths[left_idx]
        S_right = paths[right_idx]
        
        # Conditional mean and variance for the bridge
        # For log(S), the bridge has conditional distribution:
        # log(S_mid) | log(S_left), log(S_right) ~ N(mu_cond, sigma_cond^2)
        
        log_S_left = np.log(S_left)
        log_S_right = np.log(S_right)
        
        # Drift-adjusted log prices
        drift_left = log_S_left + (r - 0.5 * sigma**2) * dt_left_mid
        drift_right = log_S_right - (r - 0.5 * sigma**2) * dt_mid_right
        
        # Conditional mean (weighted average)
        weight_right = dt_left_mid / dt_left_right
        weight_left = dt_mid_right / dt_left_right
        
        mu_cond = weight_left * drift_left + weight_right * (log_S_right - (r - 0.5 * sigma**2) * dt_left_right + log_S_left)
        
        # Conditional variance
        sigma_cond = sigma * np.sqrt(dt_left_mid * dt_mid_right / dt_left_right)
        
        # Generate random increments
        Z = np.random.standard_normal(N_paths)
        
        # Simulate the midpoint
        log_S_mid = log_S_left + (r - 0.5 * sigma**2) * dt_left_mid + \
                    (dt_left_mid / dt_left_right) * (log_S_right - log_S_left - (r - 0.5 * sigma**2) * dt_left_right) + \
                    sigma_cond * Z
        
        paths[mid_idx] = np.exp(log_S_mid)
        
        # Recursively fill left and right sub-intervals
        fill_bridge(paths, left_idx, mid_idx, N_paths, r, sigma, dt)
        fill_bridge(paths, mid_idx, right_idx, N_paths, r, sigma, dt)
    
    # Fill in all intermediate points using the bridge
    fill_bridge(paths, 0, N_steps, N_paths, r, sigma, dt)
    
    return paths

def simulate_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    N_steps: int,
    N_paths: int,
    method: str = 'standard'
) -> np.ndarray:
    """
    Wrapper function to simulate stock price paths using different methods.
    
    Args:
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.
        T (float): Time to maturity (in years).
        N_steps (int): Number of time steps for the simulation.
        N_paths (int): Number of simulation paths to generate.
        method (str): Simulation method - 'standard' or 'brownian_bridge'.
                     Default is 'standard'.
    
    Returns:
        np.ndarray: A 2D numpy array of shape (N_steps + 1, N_paths)
                    containing the simulated stock price paths.
    
    Raises:
        ValueError: If method is not 'standard' or 'brownian_bridge'.
    """
    if method == 'standard':
        return simulate_gbm_paths(S0, r, sigma, T, N_steps, N_paths)
    elif method == 'brownian_bridge':
        return simulate_gbm_paths_brownian_bridge(S0, r, sigma, T, N_steps, N_paths)
    else:
        raise ValueError(f"method must be 'standard' or 'brownian_bridge', got '{method}'")

def price_american_option(
    stock_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    basis_degree: int,
    option_type: str = 'put'
) -> float:
    """
    Prices an American option using the Longstaff-Schwarz method.

    Args:
        stock_paths (np.ndarray): Simulated stock price paths.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        T (float): Time to maturity.
        basis_degree (int): Degree of Laguerre polynomial for regression.
        option_type (str): Type of option, 'put' or 'call'.

    Returns:
        float: The estimated price of the American option.
    """
    N_steps, N_paths = stock_paths.shape[0] - 1, stock_paths.shape[1]
    dt = T / N_steps

    # 1. Determine the payoff at maturity (time T)
    if option_type == 'put':
        payoff = np.maximum(K - stock_paths[-1], 0)
    elif option_type == 'call':
        payoff = np.maximum(stock_paths[-1] - K, 0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    cashflow = payoff

    # 2. Backward Induction Loop (from t_{N-1} to t_1)
    for t in range(N_steps - 1, 0, -1):
        # Discount the future cashflows back one step
        cashflow = cashflow * np.exp(-r * dt)

        S_t = stock_paths[t]
        
        # Find in-the-money paths
        if option_type == 'put':
            itm_paths_idx = np.where(K - S_t > 0)[0]
        else: # call
            itm_paths_idx = np.where(S_t - K > 0)[0]

        # If there are in-the-money paths, perform regression
        if len(itm_paths_idx) > 0:
            # Subset for regression
            X = S_t[itm_paths_idx]
            Y = cashflow[itm_paths_idx]

            # Generate basis functions
            L = laguerre_polynomials(X, basis_degree)

            # Perform least-squares regression to find coefficients
            # This corresponds to beta = (L^T L)^-1 L^T Y [cite: 284]
            beta = np.linalg.lstsq(L, Y, rcond=None)[0]

            # Estimate the continuation value
            continuation_value = L @ beta

            # Determine immediate exercise value for in-the-money paths
            if option_type == 'put':
                exercise_value = K - X
            else: # call
                exercise_value = X - K

            # Find paths where immediate exercise is optimal
            exercise_decision_mask = exercise_value > continuation_value
            exercise_idx = itm_paths_idx[exercise_decision_mask]

            # Update cashflow vector: if exercise is optimal, payoff is exercise value
            cashflow[exercise_idx] = exercise_value[exercise_decision_mask]
    
    # 3. Discount cashflows back to t=0 and compute the option price
    option_price = np.mean(cashflow * np.exp(-r * dt))
    
    return option_price

def price_and_find_boundary(
    stock_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    basis_degree: int,
    option_type: str = 'put'
):
    """
    Prices an American option and computes the exercise boundary.
    Returns both the price and the boundary points.
    """
    N_steps, N_paths = stock_paths.shape[0] - 1, stock_paths.shape[1]
    dt = T / N_steps
    
    # Set up a stock price grid for finding the boundary
    S_grid = np.linspace(min(K * 0.5, np.min(stock_paths)), max(K * 1.5, np.max(stock_paths)), 100)
    
    # Initialize boundary array (we have N_steps-1 decision points)
    exercise_boundary = np.full(N_steps, np.nan)

    if option_type == 'put':
        payoff = np.maximum(K - stock_paths[-1], 0)
    else: # call
        payoff = np.maximum(stock_paths[-1] - K, 0)
    
    cashflow = payoff

    for t in range(N_steps - 1, 0, -1):
        cashflow = cashflow * np.exp(-r * dt)
        S_t = stock_paths[t]
        
        if option_type == 'put':
            itm_paths_idx = np.where(K - S_t > 0)[0]
        else: # call
            itm_paths_idx = np.where(S_t - K > 0)[0]

        if len(itm_paths_idx) > 0:
            X = S_t[itm_paths_idx]
            Y = cashflow[itm_paths_idx]

            L = laguerre_polynomials(X, basis_degree)
            beta = np.linalg.lstsq(L, Y, rcond=None)[0]
            
            # --- Boundary Calculation ---
            # Estimate continuation values on our S_grid
            L_grid = laguerre_polynomials(S_grid, basis_degree)
            continuation_value_grid = L_grid @ beta
            
            if option_type == 'put':
                exercise_value_grid = np.maximum(K - S_grid, 0)
                # Find the first point on the grid where exercise is better
                try:
                    boundary_idx = np.where(exercise_value_grid > continuation_value_grid)[0][0]
                    exercise_boundary[t] = S_grid[boundary_idx]
                except IndexError:
                    # If exercise is never optimal, boundary is effectively 0
                    exercise_boundary[t] = 0
            else: # call
                exercise_value_grid = np.maximum(S_grid - K, 0)
                # Find the last point on the grid where exercise is better
                try:
                    boundary_idx = np.where(exercise_value_grid > continuation_value_grid)[0][-1]
                    exercise_boundary[t] = S_grid[boundary_idx]
                except IndexError:
                    # If exercise is never optimal, boundary is effectively infinity
                    exercise_boundary[t] = np.inf

            # --- Continuation of Pricing Logic (same as before) ---
            continuation_value = L @ beta

            if option_type == 'put':
                exercise_value = K - X
            else: # call
                exercise_value = X - K

            exercise_decision_mask = exercise_value > continuation_value
            exercise_idx = itm_paths_idx[exercise_decision_mask]
            cashflow[exercise_idx] = exercise_value[exercise_decision_mask]
            
    option_price = np.mean(cashflow * np.exp(-r * dt))
    
    return option_price, exercise_boundary