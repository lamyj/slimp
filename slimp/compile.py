import os
import re
import shutil
import subprocess

def compile(
        source, target_dir, optimize=True, threads=True, opencl=False,
        range_checks=True, cxxflags=None):
    """ Compile a CmdStan model
    """
    
    source_copy = os.path.join(target_dir, os.path.basename(source))
    if target_dir != os.path.dirname(source):
        shutil.copy(source, source_copy)
    subprocess.check_call(
        [
            "make", "-C", os.environ["CMDSTAN"],
            re.sub(r"\.stan$", "", os.path.abspath(str(source_copy))),
            *(["STAN_CPP_OPTIMS=TRUE"] if optimize else []),
            *(["STAN_THREADS=TRUE"] if threads else []),
            *(["STAN_OPENCL=TRUE"] if opencl else []),
            *(["STAN_NO_RANGE_CHECKS=TRUE"] if not range_checks else [])],
        env=os.environ | {"CXXFLAGS": cxxflags or ""})
    if target_dir != os.path.dirname(source):
        os.unlink(source_copy)
