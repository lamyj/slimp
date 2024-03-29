import subprocess
import sys

conda = sys.argv[1] if len(sys.argv) >= 2 else "conda"

subprocess.check_call([
    conda, "install", "--yes", "-c", "conda-forge",
    "cmake", "cmdstan", "eigen", "formulaic", "matplotlib", "ninja", "numpy",
    "pandas", "pybind11", "seaborn"])
