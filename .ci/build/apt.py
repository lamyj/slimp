import os
import subprocess

subprocess.check_call([
    "apt-get", "-y", "--no-install-recommends", "install",
    "build-essential",
    "cmake",
    "curl",
    "libboost-dev",
    "libeigen3-dev",
    "libsundials-dev",
    "libtbb-dev",
    "ninja-build",
    "pybind11-dev",
    "python3", 
    "python3-dev",
    "python3-matplotlib",
    "python3-numpy",
    "python3-pandas",
    "python3-pip",
    "python3-seaborn"])

# NOTE: formulaic is not included in Debian 12 or Ubuntu 24.04
try:
    subprocess.check_call([
        "python3", "-m", "pip", "install", "--break-system-packages", "formulaic"])
except subprocess.CalledProcessError:
    subprocess.check_call(["python3", "-m", "pip", "install", "formulaic"])

if not os.path.isdir(os.environ["CMDSTAN"]):
    os.mkdir(os.environ["CMDSTAN"])
subprocess.check_call(
    [
        "curl", "-LOJ", 
        "https://github.com/stan-dev/cmdstan/releases/download/v2.34.1/cmdstan-2.34.1.tar.gz"],
    cwd=os.environ["CMDSTAN"])
subprocess.check_call(
    ["tar", "-x", "-f", "cmdstan-2.34.1.tar.gz", "--strip-components=1"],
    cwd=os.environ["CMDSTAN"])
subprocess.check_call(["make", "-j4", "build"], cwd=os.environ["CMDSTAN"])
