import glob
import os
import shlex
import sys

import setuptools
import setuptools.command.build

sys.path.insert(0, "slimp")
import compile as slimp_compile

class BuildStanModels(setuptools.Command, setuptools.command.build.SubCommand):
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
        self.sources = list(glob.glob("slimp/*.stan"))
        self.stanc_options = shlex.split(
            os.environ.get("SLIMP_STANC_OPTIONS", ""))
        for option in ["STAN_CPP_OPTIMS", "STAN_THREADS", "STAN_NO_RANGE_CHECKS"]:
            if not any(x.startswith(f"{option}=") for x in self.stanc_options):
                self.stanc_options.append(f"{option}=TRUE")
        self.cxxflags = os.environ.get("SLIMP_CXXFLAGS", "")
        self.set_undefined_options("build_py", ("build_lib", "build_lib"))
    
    def run(self):
        for source in self.sources:
            slimp_compile.compile(
                source, os.path.join(self.build_lib, "slimp"),
                self.stanc_options, self.cxxflags)
    
    def get_source_files(self):
        return self.sources

setuptools.command.build.build.sub_commands.append(("build_stan_models", None))

setuptools.setup(
    name="slimp",
    version="0.2.0",
    
    description="Linear models with Stan and Pandas",
    
    author="Julien Lamy",
    author_email="lamy@unistra.fr",
    
    cmdclass={"build_stan_models": BuildStanModels},

    packages=["slimp"],
    
    install_requires=[
        "cmdstanpy",
        "formulaic",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn"
    ],
)
