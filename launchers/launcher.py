#!/usr/bin/env python3
"""
Height Map Generator Launcher
Choose between GUI and command-line interfaces
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description='Height Map Generator')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--cli', action='store_true', help='Run command-line interface')
    parser.add_argument('--config', type=str, help='Use specific configuration file')
    
    args = parser.parse_args()
    
    if args.gui or (not args.cli and len(sys.argv) == 1):
        # Launch GUI by default or when --gui is specified
        print("Starting Height Map Generator GUI...")
        try:
            from ui.ui import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"Error starting GUI: {e}")
            print("Make sure all required packages are installed:")
            print("pip install tkinter pillow opencv-python numpy")
            sys.exit(1)
        except Exception as e:
            print(f"GUI Error: {e}")
            sys.exit(1)
    
    elif args.cli:
        # Launch command-line interface
        print("Starting Height Map Generator CLI...")
        try:
            if args.config:
                print(f"Using configuration file: {args.config}")
                # Here you could implement config file loading
            
            import src.main_cv2 as main_cv2
            # Run the main CLI function
            from config.config import TERRAIN_CONFIG
            heightmap = main_cv2.generate_terrain_cv2(TERRAIN_CONFIG)
            if heightmap is not None:
                print("✅ Heightmap generated successfully!")
            else:
                print("❌ Failed to generate heightmap")
            
        except ImportError as e:
            print(f"Error starting CLI: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"CLI Error: {e}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
