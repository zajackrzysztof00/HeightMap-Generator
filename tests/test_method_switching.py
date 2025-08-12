#!/usr/bin/env python3
"""
Test script to verify that mountain generation method switching works properly
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
from src.ui_generator import generate_with_ui_integration
import queue

# Mock UI queues for testing
mock_message_queue = queue.Queue()
mock_progress_queue = queue.Queue()

# Test configurations with different methods
test_configs = {
    'simple': {
        'mountain_generation_method': 'simple',
        'num_mountains': 3,
        'map_size': 256,
        'map_range': [-5, 5],
        'noise_level': 0.1,
        'mountain_height_range': [1.0, 3.0],
        'mountain_width_range': [1.0, 3.0],
        'ridge_strength': 0.15,
        'valley_depth': 0.2,
    },
    'geological': {
        'mountain_generation_method': 'geological',
        'num_mountains': 3,
        'map_size': 256,
        'map_range': [-5, 5],
        'noise_level': 0.1,
        'mountain_height_range': [1.0, 3.0],
        'mountain_width_range': [1.0, 3.0],
        'ridge_strength': 0.25,
        'valley_depth': 0.35,
    }
}

# Standard configs for other components
empty_custom_mountains = {'enabled': False, 'positions': [], 'heights': [], 'widths': []}
empty_river_config = {'enabled': False, 'rivers': []}
empty_road_config = {'enabled': False, 'roads': []}
standard_smoothing = {'gaussian': {'enabled': False}, 'erosion_simulation': {'enabled': False}}
standard_export = {'filename': 'test', 'format': 'png'}
standard_advanced = {'random_seed': 42}  # Fixed seed for consistent comparison

def test_method_switching():
    """Test both mountain generation methods and compare results"""
    
    print("🧪 Testing Mountain Generation Method Switching")
    print("=" * 60)
    
    results = {}
    
    for method_name, terrain_config in test_configs.items():
        print(f"\n🏔️  Testing {method_name.upper()} method...")
        print(f"   Configuration: mountain_generation_method = '{terrain_config['mountain_generation_method']}'")
        
        try:
            # Generate terrain with this method
            X, Y, Z = generate_with_ui_integration(
                terrain_config,
                empty_custom_mountains,
                empty_river_config,
                empty_road_config,
                standard_smoothing,
                standard_export,
                standard_advanced,
                mock_message_queue,
                mock_progress_queue
            )
            
            # Store results
            results[method_name] = {
                'X': X, 'Y': Y, 'Z': Z,
                'min_elevation': np.min(Z),
                'max_elevation': np.max(Z),
                'mean_elevation': np.mean(Z),
                'std_elevation': np.std(Z)
            }
            
            print(f"   ✅ Generation successful!")
            print(f"   📊 Elevation stats:")
            print(f"      Min: {results[method_name]['min_elevation']:.3f}")
            print(f"      Max: {results[method_name]['max_elevation']:.3f}")
            print(f"      Mean: {results[method_name]['mean_elevation']:.3f}")
            print(f"      Std: {results[method_name]['std_elevation']:.3f}")
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return False
    
    # Compare results to verify they're different
    print(f"\n🔍 Comparing Methods...")
    print("-" * 40)
    
    if len(results) == 2:
        simple_z = results['simple']['Z']
        geological_z = results['geological']['Z']
        
        # Calculate difference
        diff = np.mean(np.abs(simple_z - geological_z))
        max_diff = np.max(np.abs(simple_z - geological_z))
        
        print(f"Mean absolute difference: {diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        
        # They should be different (but using same seed, so some similarity expected)
        if diff > 0.001:  # Should be significantly different
            print("✅ Methods produce DIFFERENT results - switching works!")
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Simple method
            im1 = axes[0].imshow(simple_z, cmap='terrain', origin='lower')
            axes[0].set_title('Simple Method')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[0])
            
            # Geological method
            im2 = axes[1].imshow(geological_z, cmap='terrain', origin='lower')
            axes[1].set_title('Geological Method')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[1])
            
            # Difference
            diff_map = np.abs(simple_z - geological_z)
            im3 = axes[2].imshow(diff_map, cmap='hot', origin='lower')
            axes[2].set_title('Absolute Difference')
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("📊 Comparison plot saved as 'method_comparison.png'")
            return True
        else:
            print("❌ Methods produce IDENTICAL results - switching NOT working!")
            return False
    else:
        print("❌ Could not generate both methods for comparison")
        return False

def drain_queue(q):
    """Drain a queue and return all messages"""
    messages = []
    try:
        while True:
            messages.append(q.get_nowait())
    except queue.Empty:
        pass
    return messages

if __name__ == "__main__":
    success = test_method_switching()
    
    print(f"\n{'=' * 60}")
    if success:
        print("🎉 MOUNTAIN METHOD SWITCHING TEST: PASSED")
        print("   The launcher mountain generation method selector now works correctly!")
    else:
        print("💥 MOUNTAIN METHOD SWITCHING TEST: FAILED")
        print("   The mountain generation method is still not being applied properly.")
    
    # Show any captured messages
    messages = drain_queue(mock_message_queue)
    if messages:
        print(f"\n📝 Generation Messages:")
        for msg in messages[-10:]:  # Show last 10 messages
            print(f"   {msg}")
