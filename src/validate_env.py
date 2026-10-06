import os
import sys

def validate_environment():
    """
    Automated environment, dependency, directory structure, and dataset verification script.
    """
    print("="*75)
    print("  AUTOMATED ENVIRONMENT & WORKSPACE DIAGNOSTIC CHECK")
    print("="*75)
    
    # 1. Check Python Version
    py_ver = sys.version.split()[0]
    print(f"[OK] Python Environment Version: {py_ver}")
    
    # 2. Check Package Dependencies
    required_packages = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 'joblib']
    all_packages_ok = True
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  [OK] Module '{pkg}' imported successfully.")
        except ImportError as e:
            print(f"  [FAIL] FAILED to import module '{pkg}': {e}")
            all_packages_ok = False
            
    # 3. Check Directory Structure
    required_dirs = ['data', 'src', 'outputs', 'notebooks', 'report']
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  [OK] Directory '{d}/' exists.")
        else:
            print(f"  [!] Directory '{d}/' missing, creating...")
            os.makedirs(d, exist_ok=True)
            
    # 4. Check Dataset Files
    train_path = os.path.join("data", "UNSW_NB15_training-set.csv")
    test_path = os.path.join("data", "UNSW_NB15_testing-set.csv")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_size = os.path.getsize(train_path) / (1024*1024)
        test_size = os.path.getsize(test_path) / (1024*1024)
        print(f"  [OK] Training Dataset found: {train_path} ({train_size:.2f} MB)")
        print(f"  [OK] Testing Dataset found:  {test_path} ({test_size:.2f} MB)")
    else:
        print("  [FAIL] Dataset files missing in data/ directory!")
        all_packages_ok = False
        
    print("="*75)
    if all_packages_ok:
        print("  [SUCCESS] All workspace prerequisites, dependencies, and datasets verified!")
    else:
        print("  [WARNING] Some prerequisites require attention.")
    print("="*75)

if __name__ == "__main__":
    validate_environment()
