"""
Script to run all tests.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run all tests with pytest."""
    try:
        print("🧪 Running RAG Agent Tests...")
        print("=" * 50)
        
        # Add src to Python path
        src_path = Path(__file__).parent / "src"
        
        # Run pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(src_path / "tests"),
            "-v",
            "--tb=short",
            "--color=yes"
        ], cwd=str(Path(__file__).parent))
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
