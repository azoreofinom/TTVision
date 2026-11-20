"""Run the same offline installation check used by frozen builds."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from installation_check import run_installation_check

if __name__ == "__main__":
    run_installation_check()
