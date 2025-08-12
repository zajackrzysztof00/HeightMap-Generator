#!/usr/bin/env python3
"""
Simple test runner for heightmap generator tests
"""

import sys
import os
import subprocess

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Run all tests in the tests directory"""
    print("🧪 Running Heightmap Generator Tests")
    print("=" * 50)
    
    tests_dir = os.path.join(project_root, "tests")
    
    if not os.path.exists(tests_dir):
        print("❌ Tests directory not found")
        return 1
    
    # Get all test files
    test_files = [f for f in os.listdir(tests_dir) if f.startswith("test_") and f.endswith(".py")]
    
    if not test_files:
        print("❌ No test files found")
        return 1
    
    print(f"📁 Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"   • {test_file}")
    
    print("\n🔄 Running tests...")
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        print(f"\n▶️  Running {test_file}...")
        test_path = os.path.join(tests_dir, test_file)
        
        try:
            result = subprocess.run(
                [sys.executable, test_path], 
                capture_output=True, 
                text=True, 
                cwd=project_root,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {test_file} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_file} - FAILED")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()[:200]}...")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file} - TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"❌ {test_file} - ERROR: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"🏁 SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    
    return failed

if __name__ == "__main__":
    sys.exit(main())
