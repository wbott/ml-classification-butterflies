#!/usr/bin/env python3
"""
Test runner for butterfly classification project
"""
import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run the complete test suite"""
    
    # Check if we're in the right directory
    if not Path('tests').exists():
        print("❌ Tests directory not found. Please run from project root.")
        return False
    
    print("🧪 Butterfly Classification - Test Suite")
    print("=" * 50)
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not installed. Run: pip install -r requirements-test.txt")
        return False
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not installed. Run: pip install -r requirements-test.txt")
        return False
    
    print("\n🚀 Running test suite...")
    print("-" * 30)
    
    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=tests",
        "--cov-report=term-missing",
        "--disable-warnings"
    ]
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            print("\n📊 Test Summary:")
            print("   - Model Performance: ✓")
            print("   - Data Validation: ✓") 
            print("   - Image Processing: ✓")
            print("   - Edge Cases: ✓")
            return True
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ Could not run pytest. Please check installation.")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_specific_test_category(category):
    """Run specific test category"""
    test_files = {
        'model': 'tests/test_model_performance.py',
        'data': 'tests/test_data_validation.py', 
        'image': 'tests/test_image_processing.py',
        'edge': 'tests/test_edge_cases.py'
    }
    
    if category not in test_files:
        print(f"❌ Unknown test category: {category}")
        print(f"Available categories: {list(test_files.keys())}")
        return False
    
    cmd = [sys.executable, "-m", "pytest", test_files[category], "-v"]
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {category} tests: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run butterfly classification tests")
    parser.add_argument(
        "--category", 
        choices=['model', 'data', 'image', 'edge'],
        help="Run specific test category"
    )
    parser.add_argument(
        "--fast",
        action="store_true", 
        help="Run fast tests only (skip slow integration tests)"
    )
    
    args = parser.parse_args()
    
    if args.category:
        success = run_specific_test_category(args.category)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)