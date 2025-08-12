#!/usr/bin/env python3
"""
Test script for mountain type functionality
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_mountain_types():
    """Test mountain type settings"""
    print("Testing Mountain Type Configuration:")
    print("=" * 50)
    
    # Test each mountain type
    mountain_types = ['peaked', 'ridge', 'mesa', 'volcano', 'asymmetric']
    
    for mountain_type in mountain_types:
        print(f"\nTesting mountain type: {mountain_type}")
        
        # Update the config
        from config.config import TERRAIN_CONFIG
        TERRAIN_CONFIG['mountain_type'] = mountain_type
        TERRAIN_CONFIG['num_mountains'] = 3  # Just a few for testing
        TERRAIN_CONFIG['map_size'] = 256  # Smaller for faster generation
        
        print(f"✓ Set mountain_type to: {TERRAIN_CONFIG['mountain_type']}")
        
        # Import and test the generation
        try:
            from src.ui_generator import generate_with_ui_integration
            
            # Simple message handler for testing
            def test_message_handler(msg):
                if "Creating mountain" in msg:
                    print(f"  {msg}")
            
            def test_progress_handler(progress):
                pass  # Ignore progress for test
            
            print(f"  Generating heightmap with {mountain_type} mountains...")
            result = generate_with_ui_integration(
                TERRAIN_CONFIG, 
                custom_mountains=None,
                send_message=test_message_handler,
                send_progress=test_progress_handler
            )
            
            if result is not None:
                print(f"  ✅ Generation successful for {mountain_type}")
            else:
                print(f"  ❌ Generation failed for {mountain_type}")
                
        except Exception as e:
            print(f"  ❌ Error testing {mountain_type}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Mountain type test completed!")
    print("\nTo test in UI:")
    print("1. Run: python ui.py")
    print("2. Go to Terrain tab")
    print("3. Change 'Mountain Type' dropdown")
    print("4. Click 'Generate Heightmap'")
    print("5. Check console output for mountain types")

if __name__ == "__main__":
    test_mountain_types()
