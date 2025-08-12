#!/usr/bin/env python3
"""
Development setup script for Heightmap Generator
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required Python packages"""
    packages = [
        'numpy',
        'opencv-python', 
        'Pillow',
        'tkinter'  # Usually comes with Python, but just in case
    ]
    
    print("🔧 Setting up development environment...")
    print("📦 Installing required packages...")
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"  ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {package}")
            print(f"     You may need to install it manually: pip install {package}")
        except Exception as e:
            print(f"  ⚠️  Warning installing {package}: {e}")

def setup_directories():
    """Ensure all necessary directories exist"""
    dirs = [
        'output/maps',
        'output/images', 
        'docs',
        'tests',
        'src',
        'ui',
        'config',
        'launchers'
    ]
    
    print("\n📁 Setting up project directories...")
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ {dir_path}")

def main():
    print("🏔️  Heightmap Generator - Development Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("⚠️  Warning: Python 3.7+ is recommended")
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    install_dependencies()
    
    print("\n🎉 Setup completed!")
    print("\n🚀 Quick start:")
    print("   python run_ui.py              # Start the UI")
    print("   python launchers/run_tests.py # Run tests")
    print("   python launchers/command_line_generator.py  # CLI generation")
    
    print("\n📚 Documentation available in docs/ folder")
    print("⚙️  Configuration in config/config.py")

if __name__ == "__main__":
    main()
