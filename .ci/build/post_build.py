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

python_tests_dir = os.path.join(workspace, "tests")

environment = os.environ.copy()
for name in ["DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"]:
    environment[name] = os.pathsep.join([
        lib_dir, *os.environ.get(name, "").split(os.pathsep)])
environment["PATH"] = os.pathsep.join([
    bin_dir, *os.environ.get("PATH", "").split(os.pathsep)])
environment["PYTHONPATH"] = os.pathsep.join([
    python_lib_dir, *environment.get("PYTHONPATH", "").split(os.pathsep)])

# Run unit tests
python_tests_return_code = subprocess.call(
    [sys.executable, "-m", "unittest", "discover", "-s", python_tests_dir], 
    cwd=build_dir, env=environment)

# Build and run custom model
custom_model_example = os.path.join(workspace, "custom_model_example")

custom_model_example_build = os.path.join(custom_model_example, "build")
if not os.path.isdir(custom_model_example_build):
    os.mkdir(custom_model_example_build)

python_tests_return_code = max(
    python_tests_return_code,
    subprocess.call(
        [
            "cmake", "-G", "Ninja", f"-DPython_EXECUTABLE={sys.executable}",
            custom_model_example],
        cwd=custom_model_example_build,
        env=os.environ | {
            "CXXFLAGS": f"-I{install_dir}/include",
            "LDFLAGS": f"-L{install_dir}/lib"}))
python_tests_return_code = max(
    python_tests_return_code,
    subprocess.call(
        ["cmake", "--build", ".", "--parallel", "--verbose"],
        cwd=custom_model_example_build))
python_tests_return_code = max(
    python_tests_return_code,
    subprocess.call(
        [sys.executable, "../run_model.py"], cwd=custom_model_example_build))

sys.exit(python_tests_return_code)
