#!/usr/bin/env python3
"""
Quick launcher for UI - Enhanced version with better UI
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    try:
        from ui.ui import main
        print("🏔️  Starting Heightmap Generator UI...")
        print("📁 Project root:", project_root)
        main()
    except ImportError as e:
        print(f"❌ Error importing UI: {e}")
        print("📦 Please ensure all dependencies are installed:")
        print("   pip install numpy opencv-python Pillow tkinter")
        input("Press Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running UI: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
