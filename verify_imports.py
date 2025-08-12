#!/usr/bin/env python3
"""
Import verification script - Tests all critical imports
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_import(module_path, item, description):
    """Test a specific import and report result"""
    try:
        exec(f"from {module_path} import {item}")
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description} - ImportError: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description} - Error: {e}")
        return False

def main():
    """Run all import tests"""
    print("🏔️  Heightmap Generator - Import Verification")
    print("=" * 60)
    
    tests = [
        # Core configuration
        ("config.config", "TERRAIN_CONFIG", "Configuration imports"),
        ("config.config", "CUSTOM_MOUNTAINS", "Custom mountains config"),
        
        # Core generation functions
        ("src.main_cv2", "generate_terrain_cv2", "Main terrain generation"),
        ("src.ui_generator", "generate_with_ui_integration", "UI-integrated generation"),
        
        # UI components
        ("ui.ui", "main", "Main UI function"),
        ("ui.view_terrain", "main", "Terrain viewer"),
        
        # Test modules
        ("tests.test_mountain_types", "test_config", "Mountain type tests"),
        ("tests.test_improved_geological", "test_improved_geological", "Geological tests"),
        ("tests.test_generation_methods", "compare_generation_methods", "Method comparison tests"),
        ("tests.test_method_switching", "test_method_switching", "Method switching tests"),
        
        # Launchers
        ("launchers.launcher", "main", "Main launcher"),
        ("launchers.run_tests", "main", "Test runner"),
    ]
    
    passed = 0
    failed = 0
    
    for module_path, item, description in tests:
        if test_import(module_path, item, description):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"🏁 IMPORT VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All imports working correctly!")
        print("🚀 Project is ready to use!")
    else:
        print(f"\n⚠️  {failed} import(s) failed")
        print("🔧 Check the error messages above for details")
    
    return failed

if __name__ == "__main__":
    sys.exit(main())
