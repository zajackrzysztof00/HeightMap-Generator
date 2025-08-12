#!/usr/bin/env python3
"""
Test script to demonstrate the improved geological mountain generation.
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

def test_improved_geological():
    """Test the improved geological generation with curved mountains and prominent features."""
    
    # Test configuration - focus on geological method
    test_config = TERRAIN_CONFIG.copy()
    test_config['mountain_generation_method'] = 'geological'
    test_config['num_mountains'] = 12  # More mountains for better demonstration
    test_config['map_size'] = 512     # Good size for detail visibility
    test_config['add_ridges'] = True
    test_config['add_valleys'] = True
    test_config['ridge_strength'] = 0.25  # Prominent ridges
    test_config['valley_depth'] = 0.35    # Deep valleys
    
    print("Generating terrain with IMPROVED GEOLOGICAL method...")
    print("Features:")
    print("  ✓ Curved, natural mountain boundaries (no straight lines)")
    print("  ✓ Prominent ridge systems with branching")
    print("  ✓ Deep valley networks with tributaries")
    print("  ✓ Natural mountain shapes (ridge, peaked, mesa, volcano, asymmetric)")
    print("  ✓ Realistic proportions and sizing")
    
    X, Y, Z = generate_terrain_cv2(test_config, CUSTOM_MOUNTAINS)
    
    # Create detailed visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main terrain view
    im1 = ax1.imshow(Z, cmap='terrain', origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    ax1.set_title('Improved Geological Terrain\n(Curved Mountains + Prominent Features)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    plt.colorbar(im1, ax=ax1, label='Elevation')
    
    # Contour view to show ridges and valleys
    contour_levels = np.linspace(Z.min(), Z.max(), 20)
    ax2.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.5)
    ax2.contourf(X, Y, Z, levels=contour_levels, cmap='terrain', alpha=0.7)
    ax2.set_title('Contour Lines\n(Shows Ridge and Valley Details)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    
    # 3D hillshade effect
    light = np.array([1, 1, 1])  # Light direction
    light = light / np.linalg.norm(light)
    
    # Calculate gradients
    grad_x = cv2.Sobel(Z.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(Z.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    # Calculate surface normals
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(Z)
    
    # Normalize
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm
    
    # Calculate lighting
    lighting = normal_x * light[0] + normal_y * light[1] + normal_z * light[2]
    lighting = np.clip(lighting, 0, 1)
    
    ax3.imshow(lighting, cmap='gray', origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()])
    ax3.set_title('Hillshade View\n(Shows 3D Relief and Surface Detail)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('X Coordinate')
    ax3.set_ylabel('Y Coordinate')
    
    # Elevation profile along a diagonal
    diagonal_indices = np.linspace(0, min(Z.shape) - 1, min(Z.shape)).astype(int)
    diagonal_elevations = Z[diagonal_indices, diagonal_indices]
    diagonal_distance = np.sqrt(2) * np.linspace(0, min(X.max() - X.min(), Y.max() - Y.min()), len(diagonal_elevations))
    
    ax4.plot(diagonal_distance, diagonal_elevations, 'b-', linewidth=2)
    ax4.fill_between(diagonal_distance, diagonal_elevations, alpha=0.3)
    ax4.set_title('Elevation Profile\n(Diagonal Cross-Section)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Elevation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_geological_terrain.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n=== IMPROVED GEOLOGICAL TERRAIN STATISTICS ===")
    print(f"Terrain Coverage:")
    print(f"  - Map size: {Z.shape[0]}x{Z.shape[1]} pixels")
    print(f"  - Coordinate range: {X.min():.1f} to {X.max():.1f}")
    print(f"  - Min elevation: {Z.min():.3f}")
    print(f"  - Max elevation: {Z.max():.3f}")
    print(f"  - Elevation range: {Z.max() - Z.min():.3f}")
    print(f"  - Mean elevation: {Z.mean():.3f}")
    print(f"  - Std deviation: {Z.std():.3f}")
    
    # Analyze terrain features
    high_terrain = Z > np.percentile(Z, 75)
    low_terrain = Z < np.percentile(Z, 25)
    print(f"\nTerrain Features:")
    print(f"  - High terrain (top 25%): {np.sum(high_terrain) / Z.size * 100:.1f}% of map")
    print(f"  - Low terrain (bottom 25%): {np.sum(low_terrain) / Z.size * 100:.1f}% of map")
    print(f"  - Estimated ridge coverage: ~{test_config['ridge_strength'] * 100:.0f}% intensity")
    print(f"  - Estimated valley coverage: ~{test_config['valley_depth'] * 100:.0f}% depth")
    
    # Save as grayscale heightmap
    heightmap_8bit = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)
    cv2.imwrite('improved_geological_heightmap.png', heightmap_8bit)
    
    print(f"\nFiles saved:")
    print(f"  - improved_geological_terrain.png (detailed 4-panel analysis)")
    print(f"  - improved_geological_heightmap.png (grayscale heightmap)")

if __name__ == "__main__":
    test_improved_geological()
