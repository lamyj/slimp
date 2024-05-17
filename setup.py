import glob
import os
import subprocess
import sys
import tempfile

import setuptools
import setuptools.command.build

here = os.path.abspath(os.path.dirname(__file__))

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
            *sorted(glob.glob(f"src/stan/*")),
            *sorted(glob.glob(f"src/lib/slimp/*")),
            *sorted(glob.glob(f"src/python/*.cpp"))]
        self.set_undefined_options("build_py", ("build_lib", "build_lib"))
    
    def run(self):
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
                    "cmake", "--build", build_dir,
                    "--target", "libslimp", "--target", "pyslimp",
                    "--config", "Release", "--parallel"])
    
    def get_source_files(self):
        return self.sources

setuptools.command.build.build.sub_commands.append(("build_cmake", None))

long_description = open(os.path.join(here, "README.md")).read()
setuptools.setup(
    name="slimp",
    version="0.5.0",
    
    description="Linear models with Stan and Pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    url="https://github.com/lamyj/slimp",
    
    author="Julien Lamy",
    author_email="lamy@unistra.fr",
    
    license="MIT",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: Matplotlib",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    
    keywords = [
        "statistics", "bayesian", "linear-models", "stan", "pandas",
        "matplotlib"],
    
    cmdclass={"build_cmake": BuildCMake},

    packages=setuptools.find_packages(where="src/python"),
    package_dir={"": "src/python"},
    
    install_requires=[
        "formulaic",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn"
    ],
)
