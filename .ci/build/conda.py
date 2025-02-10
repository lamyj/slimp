import subprocess
import sys

conda = sys.argv[1] if len(sys.argv) >= 2 else "conda"

subprocess.check_call([
    conda, "install", "--yes", "-c", "conda-forge",
    "boost", "cmake", "cmdstan", "eigen", "formulaic", "matplotlib", "ninja",
    "numpy", "pandas", "pybind11", "r-lme4", "r-systemfit", "rpy2", "seaborn",
    "sundials<7", "tbb", "xarray", "xtensor", "xtensor-python"])
