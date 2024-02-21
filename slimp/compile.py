import os
import re
import shutil
import subprocess

def compile(source, target_dir, stanc_options=None, cxxflags=None):
    """ Compile a CmdStan model
    """
    
    source_copy = os.path.join(target_dir, os.path.basename(source))
    if target_dir != os.path.dirname(source):
        shutil.copy(source, source_copy)
    
    subprocess.check_call(
        [
            "make", "-C", os.environ["CMDSTAN"],
            re.sub(r"\.stan$", "", os.path.abspath(str(source_copy))),
            *(stanc_options or [])],
        env=os.environ | {"CXXFLAGS": cxxflags or ""})
    if target_dir != os.path.dirname(source):
        os.unlink(source_copy)
