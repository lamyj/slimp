import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap

def compile(stan_file, h_file, cpp_file, include_path=None):
    name = os.path.splitext(os.path.basename(stan_file))[0]
    
    header = textwrap.dedent(f"""\
        #include <stan/model/model_header.hpp>
        
        stan::model::model_base & new_{name}(
            stan::io::var_context & data_context, unsigned int seed,
            std::ostream * msg_stream);
    """)
    with open(h_file, "w") as fd:
        fd.write(header)

    subprocess.check_call([
        f"{os.environ['CMDSTAN']}/bin/stanc",
        *(["--include-paths", include_path] if include_path else []),
        stan_file, "--name", "model", "--O1",
        "--o", cpp_file])
    
    with open(cpp_file) as fd:
        contents = fd.read()
    
    contents = re.sub("model_namespace", name, contents)
    contents = re.sub("using stan_model = [^;]+", "", contents)
    contents = re.sub("stan_model", f"{name}::model", contents)
    contents = re.sub("new_model", f"new_{name}", contents)
    contents = re.sub(
        "get_stan_profile_data", f"get_stan_profile_data_{name}", contents)
    
    with open(cpp_file, "w") as fd:
        fd.write(contents)

def stan_info(names, optimize=True, use_threads=1, range_checks=False):
    if isinstance(names, str):
        names = [names]
    
    makefile = textwrap.dedent(f"""\
        STAN = $(CMDSTAN)/stan/
        STAN_CPP_OPTIMS = {int(optimize)}
        STAN_THREADS = {int(use_threads)}
        STAN_NO_RANGE_CHECKS = {(int(not range_checks))}

        include $(CMDSTAN)/makefile

        cxxflags:
            @echo $(CPPFLAGS) $(CXXFLAGS)
        ldflags:
            @echo $(LDFLAGS) | sed 's/-Wl,-L,/-L/g'
        libs:
            @echo $(LDLIBS)
    """)
    makefile = re.sub("    ", "\t", makefile)
    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "makefile")
        with open(path, "w") as fd:
            fd.write(makefile)
        flags = subprocess.check_output([
            "make", "-I", os.environ["CMDSTAN"], "-f", path, *names])
    return shlex.split(flags.decode())

def main():
    try:
        command = sys.argv[1]
    except IndexError:
        print("Missing command")
        return 1
    
    if command == "compile":
        parser = argparse.ArgumentParser()
        parser.add_argument("stan_file")
        parser.add_argument("h_file")
        parser.add_argument("cpp_file")
        parser.add_argument("--include-path", "-I")
        arguments = parser.parse_args(sys.argv[2:])
        compile(**vars(arguments))
    elif command == "info":
        parser = argparse.ArgumentParser()
        parser.add_argument("names", metavar="name", nargs="+")
        arguments = parser.parse_args(sys.argv[2:])
        print(shlex.join(stan_info(**vars(arguments))))
    else:
        print(f"Unknown command: {command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
