#!/usr/bin/env python3
"""
Command line launcher for direct terrain generation
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    try:
        from config.config import TERRAIN_CONFIG
        from src.main_cv2 import generate_terrain_cv2
        
        print("🏔️  Command Line Heightmap Generator")
        print("=" * 50)
        
        # Show current config
        print("Current configuration:")
        for key, value in TERRAIN_CONFIG.items():
            print(f"  {key}: {value}")
        
        print("\n🔄 Generating heightmap...")
        
        # Generate using current config
        heightmap = generate_terrain_cv2(TERRAIN_CONFIG)
        
        if heightmap is not None:
            print("✅ Heightmap generated successfully!")
            print(f"📏 Size: {heightmap.shape}")
            print(f"📁 Saved to: output/maps/")
        else:
            print("❌ Failed to generate heightmap")
            
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("📦 Please ensure all dependencies are installed")
        return 1
    except Exception as e:
        print(f"❌ Error generating heightmap: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
