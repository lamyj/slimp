import os
import re
import subprocess

def compile(
        source, target, optimize=False, opencl=False, range_checks=True,
        cxxflags=None):
    """ Compile a CmdStan model
    """
    
    subprocess.check_call(
        [
            "make", "-C", os.environ["CMDSTAN"],
            re.sub(r"\.stan$", "", os.path.abspath(str(source))),
            *(["STAN_CPP_OPTIMS=TRUE"] if optimize else []),
            *(["STAN_OPENCL=TRUE"] if opencl else []),
            *(["STAN_NO_RANGE_CHECKS=TRUE"] if not range_checks else [])],
        env=os.environ | {"CXXFLAGS": cxxflags or ""})
    # FIXME: move to target
