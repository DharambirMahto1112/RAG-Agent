"""
Script to run the Streamlit UI.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application."""
    try:
        # Path to the Streamlit app
        app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
        
        print("🚀 Starting RAG Agent Streamlit UI...")
        print(f"📁 App path: {app_path}")
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped!")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
