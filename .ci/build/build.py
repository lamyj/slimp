import multiprocessing
import os
import re
import subprocess
import sys

workspace = os.environ["WORKSPACE"]
build_dir = os.environ.get("BUILD_DIR", os.path.join(workspace, "build"))
install_dir = os.environ.get("INSTALL_DIR", os.path.join(workspace, "install"))

for dir in [build_dir, install_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

subprocess.check_call(
    [
        "cmake", "-G", "Ninja", f"-DPython_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={install_dir}", 
        workspace],
    cwd=build_dir,
    env=os.environ|{
        "STAN": f"{os.environ['CMDSTAN']}/stan/",
        "STAN_THREADS":"1", "STAN_NO_RANGE_CHECKS":"1", "CXX":"g++"})

subprocess.check_call(
    [
        "cmake", "--build", ".", "--target", "install", "--config", "Release",
        "--parallel"],
    cwd=build_dir)
