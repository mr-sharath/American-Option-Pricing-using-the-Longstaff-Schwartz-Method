# File: src/basis_functions.py

import numpy as np

def laguerre_polynomials(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Generates a matrix of Laguerre polynomials up to a given degree.

    Args:
        x (np.ndarray): Input array of stock prices.
        degree (int): The maximum degree of the polynomial to generate.

    Returns:
        np.ndarray: A matrix where each column is a Laguerre polynomial
                    of a certain degree evaluated at x. The shape is
                    (len(x), degree + 1).
    """
    n = len(x)
    # The matrix L will hold the polynomial values. Column j for degree j.
    L = np.zeros((n, degree + 1))

    # L_0(x) = 1
    L[:, 0] = 1
    
    if degree > 0:
        # L_1(x) = 1 - x
        L[:, 1] = 1 - x
    
    if degree > 1:
        # L_n(x) = ((2n - 1 - x) * L_{n-1}(x) - (n - 1) * L_{n-2}(x)) / n
        for i in range(2, degree + 1):
            L[:, i] = ((2 * i - 1 - x) * L[:, i - 1] - (i - 1) * L[:, i - 2]) / i
            
    return L