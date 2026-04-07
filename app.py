"""
app.py — HuggingFace Spaces entry point.
Delegates to the FastAPI server in api/server.py.
HF Spaces with Docker SDK will pick up the CMD in Dockerfile,
but this file also allows direct `python app.py` startup.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "api.server:app",
            "--host", "0.0.0.0",
            "--port", "7860",
            "--workers", "1",
        ],
        check=True,
    )
