#!/usr/bin/env python3
"""
Main launcher for Heightmap Generator UI
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    try:
        from ui.ui import main
        main()
    except ImportError as e:
        print(f"Error importing UI: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install numpy opencv-python Pillow tkinter")
        sys.exit(1)
    except Exception as e:
        print(f"Error running UI: {e}")
        sys.exit(1)
