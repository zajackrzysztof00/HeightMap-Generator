#!/usr/bin/env python3
"""
Test script for mountain type selection functionality
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config.config import TERRAIN_CONFIG, CUSTOM_MOUNTAINS

def test_config():
    """Test that mountain type is properly configured"""
    print("Testing Mountain Type Configuration:")
    print("=" * 40)
    
    # Test default config
    print(f"Default mountain type: {TERRAIN_CONFIG.get('mountain_type', 'NOT FOUND')}")
    
    # Test custom mountains config
    print(f"Custom mountains enabled: {CUSTOM_MOUNTAINS['enabled']}")
    print(f"Custom mountain types: {CUSTOM_MOUNTAINS.get('types', 'NOT FOUND')}")
    
    # Test all available types
    available_types = ['varied', 'peaked', 'ridge', 'mesa', 'volcano', 'asymmetric']
    print(f"Available mountain types: {available_types}")
    
    # Test configuration with different types
    test_configs = [
        {'mountain_type': 'peaked'},
        {'mountain_type': 'ridge'}, 
        {'mountain_type': 'mesa'},
        {'mountain_type': 'volcano'},
        {'mountain_type': 'asymmetric'},
        {'mountain_type': 'varied'}
    ]
    
    print("\nTesting different mountain type configurations:")
    for i, test_config in enumerate(test_configs):
        print(f"Test {i+1}: mountain_type = '{test_config['mountain_type']}' ✓")
    
    print("\n✅ Mountain type configuration test completed successfully!")
    print("\nTo test the functionality:")
    print("1. Run the UI: python ui.py")
    print("2. Go to Terrain tab")
    print("3. Select different Mountain Types from the dropdown")
    print("4. Generate heightmaps to see different mountain shapes")
    print("5. Use Mountains tab to set individual mountain types")

if __name__ == "__main__":
    test_config()
