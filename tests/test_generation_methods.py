#!/usr/bin/env python3
"""
Test script to compare simple vs geological mountain generation methods.
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.main_cv2 import generate_terrain_cv2
from config.config import TERRAIN_CONFIG, CUSTOM_MOUNTAINS

def compare_generation_methods():
    """Compare the two generation methods side by side."""
    
    # Test configuration
    test_config = TERRAIN_CONFIG.copy()
    test_config['num_mountains'] = 8  # Fewer mountains for clearer comparison
    test_config['map_size'] = 512     # Smaller for faster generation
    
    print("Generating terrain with SIMPLE method...")
    test_config['mountain_generation_method'] = 'simple'
    X1, Y1, Z1 = generate_terrain_cv2(test_config, CUSTOM_MOUNTAINS)
    
    print("Generating terrain with GEOLOGICAL method...")
    test_config['mountain_generation_method'] = 'geological'
    X2, Y2, Z2 = generate_terrain_cv2(test_config, CUSTOM_MOUNTAINS)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Simple method
    im1 = ax1.imshow(Z1, cmap='terrain', origin='lower', extent=[X1.min(), X1.max(), Y1.min(), Y1.max()])
    ax1.set_title('Simple Generation Method\n(Original/Basic Shapes)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    plt.colorbar(im1, ax=ax1, label='Elevation')
    
    # Geological method
    im2 = ax2.imshow(Z2, cmap='terrain', origin='lower', extent=[X2.min(), X2.max(), Y2.min(), Y2.max()])
    ax2.set_title('Geological Generation Method\n(Realistic Features)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    plt.colorbar(im2, ax=ax2, label='Elevation')
    
    plt.tight_layout()
    plt.savefig('generation_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n=== COMPARISON STATISTICS ===")
    print(f"Simple Method:")
    print(f"  - Min elevation: {Z1.min():.3f}")
    print(f"  - Max elevation: {Z1.max():.3f}")
    print(f"  - Mean elevation: {Z1.mean():.3f}")
    print(f"  - Std deviation: {Z1.std():.3f}")
    
    print(f"\nGeological Method:")
    print(f"  - Min elevation: {Z2.min():.3f}")
    print(f"  - Max elevation: {Z2.max():.3f}")
    print(f"  - Mean elevation: {Z2.mean():.3f}")
    print(f"  - Std deviation: {Z2.std():.3f}")
    
    # Save individual images
    cv2.imwrite('simple_method.png', ((Z1 - Z1.min()) / (Z1.max() - Z1.min()) * 255).astype(np.uint8))
    cv2.imwrite('geological_method.png', ((Z2 - Z2.min()) / (Z2.max() - Z2.min()) * 255).astype(np.uint8))
    
    print(f"\nFiles saved:")
    print(f"  - generation_method_comparison.png (side-by-side comparison)")
    print(f"  - simple_method.png (grayscale heightmap)")
    print(f"  - geological_method.png (grayscale heightmap)")

if __name__ == "__main__":
    compare_generation_methods()
