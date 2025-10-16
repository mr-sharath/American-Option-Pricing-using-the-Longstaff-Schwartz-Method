"""
Main interface for American Option Pricing.

This module provides a simple, unified interface for pricing American options
using the Longstaff-Schwartz Method with different simulation techniques.

Example:
    >>> from american_option_pricer import AmericanOptionPricer
    >>> 
    >>> pricer = AmericanOptionPricer(
    ...     S0=100, K=100, r=0.03, sigma=0.15, T=1.0,
    ...     N_steps=100, N_paths=100000
    ... )
    >>> 
    >>> price = pricer.price(option_type='put', method='brownian_bridge')
    >>> print(f"American Put Price: ${price:.4f}")
"""

import numpy as np
from simulation import simulate_standard_gbm, simulate_brownian_bridge
from pricing import price_american_option, price_with_boundary
from benchmarks import validate_against_benchmark


class AmericanOptionPricer:
    """
    Unified interface for American option pricing using LSM method.
    
    This class encapsulates all the functionality needed to price American
    options, including simulation, pricing, and validation.
    
    Attributes:
        S0 (float): Initial stock price.
        K (float): Strike price.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        T (float): Time to maturity.
        N_steps (int): Number of time steps.
        N_paths (int): Number of simulation paths.
        basis_degree (int): Degree of Laguerre polynomials.
    """
    
    def __init__(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        N_steps: int = 100,
        N_paths: int = 100000,
        basis_degree: int = 3
    ):
        """
        Initialize the American Option Pricer.
        
        Args:
            S0: Initial stock price
            K: Strike price
            r: Risk-free interest rate
            sigma: Volatility
            T: Time to maturity (years)
            N_steps: Number of time steps (default: 100)
            N_paths: Number of Monte Carlo paths (default: 100000)
            basis_degree: Degree of Laguerre polynomials (default: 3)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N_steps = N_steps
        self.N_paths = N_paths
        self.basis_degree = basis_degree
        
        self._paths = None
        self._last_method = None
    
    def simulate_paths(self, method: str = 'brownian_bridge', seed: int = None) -> np.ndarray:
        """
        Simulate stock price paths.
        
        Args:
            method: Simulation method ('standard' or 'brownian_bridge')
            seed: Random seed for reproducibility
        
        Returns:
            np.ndarray: Simulated paths of shape (N_steps + 1, N_paths)
        """
        if method == 'standard':
            self._paths = simulate_standard_gbm(
                self.S0, self.r, self.sigma, self.T,
                self.N_steps, self.N_paths, seed
            )
        elif method == 'brownian_bridge':
            self._paths = simulate_brownian_bridge(
                self.S0, self.r, self.sigma, self.T,
                self.N_steps, self.N_paths, seed
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'standard' or 'brownian_bridge'")
        
        self._last_method = method
        return self._paths
    
    def price(
        self,
        option_type: str = 'put',
        method: str = 'brownian_bridge',
        seed: int = None,
        use_cached_paths: bool = False
    ) -> float:
        """
        Price an American option.
        
        Args:
            option_type: 'put' or 'call'
            method: Simulation method ('standard' or 'brownian_bridge')
            seed: Random seed
            use_cached_paths: If True, use previously simulated paths
        
        Returns:
            float: Option price
        """
        if not use_cached_paths or self._paths is None or self._last_method != method:
            self.simulate_paths(method, seed)
        
        price = price_american_option(
            self._paths, self.K, self.r, self.T,
            self.basis_degree, option_type
        )
        
        return price
    
    def price_with_exercise_boundary(
        self,
        option_type: str = 'put',
        method: str = 'brownian_bridge',
        seed: int = None
    ) -> tuple:
        """
        Price an American option and compute exercise boundary.
        
        Args:
            option_type: 'put' or 'call'
            method: Simulation method
            seed: Random seed
        
        Returns:
            tuple: (price, exercise_boundary)
        """
        self.simulate_paths(method, seed)
        
        price, boundary = price_with_boundary(
            self._paths, self.K, self.r, self.T,
            self.basis_degree, option_type
        )
        
        return price, boundary
    
    def compare_methods(self, option_type: str = 'put', seed: int = None) -> dict:
        """
        Compare standard GBM vs Brownian Bridge methods.
        
        Args:
            option_type: 'put' or 'call'
            seed: Random seed
        
        Returns:
            dict: Comparison results
        """
        # Price with standard GBM
        price_std = self.price(option_type, 'standard', seed)
        
        # Price with Brownian Bridge
        price_bridge = self.price(option_type, 'brownian_bridge', seed)
        
        return {
            'standard_gbm': price_std,
            'brownian_bridge': price_bridge,
            'difference': abs(price_std - price_bridge),
            'relative_diff_pct': abs(price_std - price_bridge) / price_std * 100
        }
    
    def validate(
        self,
        option_type: str = 'put',
        method: str = 'brownian_bridge',
        benchmark_name: str = 'standard_case',
        seed: int = None
    ) -> dict:
        """
        Price option and validate against benchmark.
        
        Args:
            option_type: 'put' or 'call'
            method: Simulation method
            benchmark_name: Name of benchmark case
            seed: Random seed
        
        Returns:
            dict: Validation results
        """
        price = self.price(option_type, method, seed)
        
        result = validate_against_benchmark(
            price, benchmark_name, f'american_{option_type}'
        )
        
        return result
    
    def get_parameters(self) -> dict:
        """Get all pricing parameters as a dictionary."""
        return {
            'S0': self.S0,
            'K': self.K,
            'r': self.r,
            'sigma': self.sigma,
            'T': self.T,
            'N_steps': self.N_steps,
            'N_paths': self.N_paths,
            'basis_degree': self.basis_degree
        }
    
    def __repr__(self) -> str:
        """String representation of the pricer."""
        return (
            f"AmericanOptionPricer(S0={self.S0}, K={self.K}, r={self.r}, "
            f"sigma={self.sigma}, T={self.T}, N_steps={self.N_steps}, "
            f"N_paths={self.N_paths}, basis_degree={self.basis_degree})"
        )


def quick_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'put',
    method: str = 'brownian_bridge',
    N_steps: int = 100,
    N_paths: int = 100000,
    seed: int = None
) -> float:
    """
    Quick one-line function to price an American option.
    
    Args:
        S0: Initial stock price
        K: Strike price
        r: Risk-free rate
        sigma: Volatility
        T: Time to maturity
        option_type: 'put' or 'call'
        method: 'standard' or 'brownian_bridge'
        N_steps: Number of time steps
        N_paths: Number of paths
        seed: Random seed
    
    Returns:
        float: Option price
    
    Example:
        >>> price = quick_price(100, 100, 0.03, 0.15, 1.0, 'put')
        >>> print(f"Price: ${price:.4f}")
    """
    pricer = AmericanOptionPricer(S0, K, r, sigma, T, N_steps, N_paths)
    return pricer.price(option_type, method, seed)
