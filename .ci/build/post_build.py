import os
import subprocess
import sys
import sysconfig

workspace = os.environ["WORKSPACE"]
build_dir = os.environ.get("BUILD_DIR", os.path.join(workspace, "build"))
install_dir = os.environ.get("INSTALL_DIR", os.path.join(workspace, "install"))

bin_dir = os.path.join(install_dir, "bin")
lib_dir = os.path.join(install_dir, "lib")
python_lib_dir = os.path.join(
    install_dir,
    sysconfig.get_path(
        "purelib", {"posix":"posix_prefix", "nt":"nt"}[os.name], {"base": "."}))

python_tests_dir = os.path.join(workspace, "tests", "python")

# C++ tests: only library path is needed
environment = os.environ.copy()
for name in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"]:
    environment[name] = os.pathsep.join([
        lib_dir, *os.environ.get(name, "").split(os.pathsep)])
environment["PATH"] = os.pathsep.join([
    bin_dir, *os.environ.get("PATH", "").split(os.pathsep)])
environment["PYTHONPATH"] = os.pathsep.join([
    python_lib_dir, *environment.get("PYTHONPATH", "").split(os.pathsep)])

python_tests_return_code = subprocess.call(
    [sys.executable, "-m", "unittest", "discover", "-s", python_tests_dir], 
    cwd=build_dir, env=environment)

sys.exit(python_tests_return_code)
