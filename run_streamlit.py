"""
Script to run the Streamlit UI in a production-friendly way.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path

def _find_free_port(preferred: int) -> int:
    """Return preferred if free, else find an available ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", preferred))
            return preferred
        except OSError:
            pass
    # Ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    """Run the Streamlit application."""
    try:
        # Path to the Streamlit app
        app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
        
        # Resolve port/address from env with safe defaults
        env_port = os.environ.get("PORT") or os.environ.get("STREAMLIT_SERVER_PORT") or "8501"
        try:
            preferred_port = int(env_port)
        except ValueError:
            preferred_port = 8501
        port = _find_free_port(preferred_port)

        address = os.environ.get("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
        headless = os.environ.get("STREAMLIT_HEADLESS", "true").lower() in {"1", "true", "yes"}

        print("Starting RAG Agent Streamlit UI...")
        print(f"App path: {app_path}")
        print(f"Binding to {address}:{port} (headless={headless})")

        # Build streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", str(port),
            "--server.address", address,
            "--server.headless", "true" if headless else "false",
            "--browser.gatherUsageStats", "false"
        ]

        # Run Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nStreamlit app stopped!")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
