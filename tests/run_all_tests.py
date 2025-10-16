"""
Master test runner.

Runs all unit tests and benchmark validations.
"""

import sys
import time

# Import all test modules
import test_simulation
import test_pricing
import test_benchmarks


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 20 + "AMERICAN OPTION PRICER")
    print(" " * 20 + "COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Run simulation tests
        test_simulation.run_all_tests()
        
        # Run pricing tests
        test_pricing.run_all_tests()
        
        # Run benchmark tests
        test_benchmarks.run_all_tests()
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print("=" * 70 + "\n")
        
        return 0
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print("=" * 70 + "\n")
        return 1
    
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ UNEXPECTED ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        print("=" * 70 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
