import glob
import os
import subprocess
import sys
import tempfile

import setuptools
import setuptools.command.build

class BuildCMake(setuptools.Command, setuptools.command.build.SubCommand):
    def __init__(self, *args, **kwargs):
        setuptools.Command.__init__(self, *args, **kwargs)
        setuptools.command.build.SubCommand.__init__(self, *args, **kwargs)
        self.build_lib = None
        self.editable_mode = False
        
        self.sources = []
        self.stanc_options = ""
        self.cxxflags = ""
    
    def initialize_options(self):
        pass
        
    def finalize_options(self):
        self.sources = [
            *sorted(glob.glob(f"slimp/*.stan")),
            *sorted(glob.glob(f"slimp/*.h")),
            *sorted(glob.glob(f"slimp/*.cpp"))]
        self.set_undefined_options("build_py", ("build_lib", "build_lib"))
    
    def run(self):
        here = os.path.abspath(os.path.dirname(__file__))
        with tempfile.TemporaryDirectory() as build_dir:
            subprocess.check_call(
                [
                    "cmake", f"-DPython_EXECUTABLE={sys.executable}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY="
                        f"{os.path.join(here, self.build_lib, 'slimp')}", 
                    "-S", here, "-B", build_dir])
            
            subprocess.check_call(
                [
                    "cmake", "--build", build_dir, "--target", "_slimp",
                    "--config", "Release", "--parallel"])
    
    def get_source_files(self):
        return self.sources

setuptools.command.build.build.sub_commands.append(("build_cmake", None))

setuptools.setup(
    name="slimp",
    version="0.3.0",
    
    description="Linear models with Stan and Pandas",
    
    author="Julien Lamy",
    author_email="lamy@unistra.fr",
    
    cmdclass={"build_cmake": BuildCMake},

    packages=["slimp"],
    
    install_requires=[
        "formulaic",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn"
    ],
)
