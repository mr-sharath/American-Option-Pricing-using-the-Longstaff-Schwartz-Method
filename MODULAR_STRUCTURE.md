## Modular Code Structure

The codebase has been restructured into clean, separate modules for better organization, testing, and understanding.

## Directory Structure

```
ams514_project1/
├── src/
│   ├── simulation/                    # Stock price simulation modules
│   │   ├── __init__.py
│   │   ├── standard_gbm.py           # Standard GBM simulation
│   │   └── brownian_bridge.py        # Brownian Bridge simulation
│   │
│   ├── pricing/                       # LSM pricing algorithm
│   │   ├── __init__.py
│   │   └── lsm_algorithm.py          # Longstaff-Schwartz method
│   │
│   ├── benchmarks/                    # Validation benchmarks
│   │   ├── __init__.py
│   │   └── gustafsson_benchmarks.py  # Gustafsson (2015) benchmarks
│   │
│   ├── basis_functions.py            # Laguerre polynomials
│   ├── american_option_pricer.py     # Main unified interface
│   └── lsm_pricer.py                 # Legacy monolithic file (deprecated)
│
├── tests/                             # Comprehensive test suite
│   ├── test_simulation.py            # Tests for simulation modules
│   ├── test_pricing.py               # Tests for pricing algorithm
│   ├── test_benchmarks.py            # Benchmark validation tests
│   └── run_all_tests.py              # Master test runner
│
├── examples/                          # Usage examples
│   ├── example_basic_usage.py        # Basic pricing examples
│   ├── example_method_comparison.py  # Compare simulation methods
│   └── example_exercise_boundary.py  # Exercise boundary computation
│
├── notebooks/                         # Jupyter notebooks (original)
├── results/                           # Output figures and results
└── report/                            # Project report
```

## Module Descriptions

### 1. Simulation Modules (`src/simulation/`)

#### `standard_gbm.py`
- **Purpose**: Standard sequential GBM simulation
- **Key Function**: `simulate_standard_gbm()`
- **Reference**: Gustafsson (2015), equation 1.5
- **Use Case**: Baseline simulation method

#### `brownian_bridge.py`
- **Purpose**: Brownian Bridge variance reduction technique
- **Key Function**: `simulate_brownian_bridge()`
- **Reference**: Gustafsson (2015), Glasserman (2003)
- **Use Case**: Improved accuracy with same computational cost

### 2. Pricing Module (`src/pricing/`)

#### `lsm_algorithm.py`
- **Purpose**: Longstaff-Schwartz Method implementation
- **Key Functions**:
  - `price_american_option()`: Basic pricing
  - `price_with_boundary()`: Pricing + exercise boundary
- **Reference**: Longstaff & Schwartz (2001), Gustafsson (2015)
- **Algorithm**: Backward induction with least-squares regression

### 3. Benchmarks Module (`src/benchmarks/`)

#### `gustafsson_benchmarks.py`
- **Purpose**: Validation against academic benchmarks
- **Key Functions**:
  - `validate_against_benchmark()`: Compare with Gustafsson values
  - `compare_methods()`: Compare simulation methods
  - `print_validation_report()`: Formatted output
- **Benchmarks**: Gustafsson (2015) Table 3.1

### 4. Main Interface (`src/american_option_pricer.py`)

#### `AmericanOptionPricer` Class
- **Purpose**: Unified, user-friendly interface
- **Features**:
  - Simple initialization with parameters
  - Method comparison
  - Automatic validation
  - Exercise boundary computation
- **Example**:
  ```python
  pricer = AmericanOptionPricer(S0=100, K=100, r=0.03, sigma=0.15, T=1.0)
  price = pricer.price(option_type='put', method='brownian_bridge')
  ```

## Usage Guide

### Quick Start

```python
from american_option_pricer import quick_price

# One-liner pricing
price = quick_price(S0=100, K=100, r=0.03, sigma=0.15, T=1.0, option_type='put')
print(f"American Put: ${price:.4f}")
```

### Using Individual Modules

```python
# Import specific modules
from simulation import simulate_brownian_bridge
from pricing import price_american_option

# Simulate paths
paths = simulate_brownian_bridge(
    S0=100, r=0.03, sigma=0.15, T=1.0,
    N_steps=100, N_paths=100000, seed=42
)

# Price option
price = price_american_option(
    paths, K=100, r=0.03, T=1.0,
    basis_degree=3, option_type='put'
)
```

### Running Tests

```bash
# Run all tests
cd tests
python run_all_tests.py

# Run specific test modules
python test_simulation.py
python test_pricing.py
python test_benchmarks.py
```

### Running Examples

```bash
cd examples
python example_basic_usage.py
python example_method_comparison.py
python example_exercise_boundary.py
```

## Benefits of Modular Structure

### 1. **Separation of Concerns**
- Each module has a single, well-defined responsibility
- Simulation logic separate from pricing logic
- Benchmarks isolated for easy updates

### 2. **Easy Testing**
- Each module can be tested independently
- Comprehensive test coverage
- Clear validation against benchmarks

### 3. **Better Understanding**
- Code is organized by functionality
- Clear documentation in each module
- Easy to find and understand specific algorithms

### 4. **Maintainability**
- Changes to one module don't affect others
- Easy to add new simulation methods
- Simple to update benchmarks

### 5. **Reusability**
- Modules can be imported independently
- Functions can be used in different contexts
- Easy to extend for new projects

## Implementation Details

### Standard GBM
```
S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
```
- Sequential forward simulation
- Independent random increments
- Euler discretization

### Brownian Bridge
```
1. Generate S(T) directly
2. Recursively fill intermediate points
3. Use conditional distributions
```
- Variance reduction technique
- Better convergence properties
- Same computational cost

### LSM Algorithm
```
1. Initialize payoffs at maturity
2. Backward induction:
   - Discount cashflows
   - Regress on in-the-money paths
   - Compare exercise vs continuation
   - Update cashflows
3. Discount to t=0
```
- Laguerre polynomials (degree 3)
- Least-squares regression
- Optimal exercise decisions

## Validation Results

| Method           | Price   | Error vs Benchmark | Status |
|------------------|---------|-------------------|--------|
| Standard GBM     | 4.8140  | 0.0066 (0.14%)   | ✅ PASS |
| Brownian Bridge  | 4.8178  | 0.0028 (0.06%)   | ✅ PASS |
| Gustafsson Benchmark | 4.8206 | -            | -      |

**Improvement**: Brownian Bridge shows 58% error reduction

## References

1. **Longstaff & Schwartz (2001)**: "Valuing American Options by Simulation"
2. **Gustafsson (2015)**: "Evaluating the Longstaff-Schwartz method"
3. **Glasserman (2003)**: "Monte Carlo Methods in Financial Engineering"

## Migration from Old Structure

If you were using the old `lsm_pricer.py` file:

### Old Way:
```python
import lsm_pricer
paths = lsm_pricer.simulate_gbm_paths(...)
price = lsm_pricer.price_american_option(...)
```

### New Way:
```python
from american_option_pricer import AmericanOptionPricer
pricer = AmericanOptionPricer(...)
price = pricer.price(...)
```

The old `lsm_pricer.py` file is still available for backward compatibility but is deprecated.

## Future Enhancements

Potential additions to the modular structure:
- [ ] Additional simulation methods (antithetic variates, control variates)
- [ ] More basis functions (Hermite, Chebyshev polynomials)
- [ ] Dividend-paying stocks
- [ ] Greeks computation
- [ ] Multi-asset options
- [ ] Additional benchmarks (Binomial tree, Finite difference)
