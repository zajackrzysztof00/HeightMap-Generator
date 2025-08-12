#!/usr/bin/env python3
"""
Simple script to generate and display terrain with enhanced roads and rivers
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
from src.main_cv2 import generate_terrain_cv2, create_height_map_cv2

def main():
    print("=== Terrain Viewer (Quick Preview) ===")
    print("Generating quick preview terrain (256x256) with enhanced roads and rivers...")
    
    # Import the configuration to modify it temporarily
    from config.config import TERRAIN_CONFIG
    
    # Create a quick preview configuration
    preview_config = TERRAIN_CONFIG.copy()
    preview_config.update({
        'map_size': 256,           # Quick preview size
        'noise_level': 0.06,       # Slightly less noise for cleaner preview
    })
    
    print(f"Preview configuration: {preview_config['num_mountains']} mountains, {preview_config['map_size']}x{preview_config['map_size']} resolution")
    
    # Generate terrain with preview configuration
    X, Y, Z = generate_terrain_cv2(config=preview_config)
    
    # Create a more detailed view
    plt.figure(figsize=(10, 8))  # Slightly smaller figure for preview
    
    # Create the terrain plot
    terrain_plot = plt.contourf(X, Y, Z, levels=30, cmap='terrain', alpha=0.9)  # Fewer levels for preview
    plt.colorbar(terrain_plot, label='Elevation')
    
    # Add contour lines for better depth perception
    plt.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)  # Fewer contour lines
    
    plt.title('Quick Preview: Enhanced Terrain with Roads and Rivers\n(256x256 - Roads: wider cuts, Rivers: flat water surfaces)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Add grid for reference
    plt.grid(True, alpha=0.3)
    
    # Make sure the aspect ratio is equal
    plt.axis('equal')
    
    # Show statistics
    print(f"Preview terrain elevation range: {Z.min():.3f} to {Z.max():.3f}")
    print(f"Preview terrain size: {Z.shape[0]} x {Z.shape[1]} pixels")
    print(f"Generation time: Much faster with 256x256 vs 1025x1025!")
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuick preview terrain generated successfully!")
    print("Roads should be clearly visible with enhanced features:")
    print("- Increased width (0.08-0.12)")
    print("- Deeper cuts (0.15-0.25)")
    print("- Enhanced banking/embankments")
    print("- Sharper edge profiles")
    print("\nFor full resolution (1025x1025), run: python main_cv2.py")

if __name__ == "__main__":
    main()
