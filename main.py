import subprocess, sys, pathlib

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend.py"], cwd=here)
