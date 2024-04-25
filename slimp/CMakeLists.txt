find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
        -c "import os; import sysconfig; \
            scheme = {'posix': 'posix_prefix', 'nt': 'nt'}[os.name]; \
            path = sysconfig.get_path('purelib', scheme, {'base': '.'}); \
            print(path.replace(os.path.sep, '/'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(compile_helper ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compile.py)

execute_process(
    COMMAND ${compile_helper} info cxxflags
    OUTPUT_VARIABLE STAN_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND ${compile_helper} info ldflags
    OUTPUT_VARIABLE STAN_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
    COMMAND ${compile_helper} info libs
    OUTPUT_VARIABLE STAN_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

file(GLOB_RECURSE header_files "*.h")
file(GLOB_RECURSE source_files "*.cpp")
file(GLOB_RECURSE python_files "*.py")
file(
    GLOB_RECURSE stan_files
    "*_sampler.stan" "*_log_likelihood.stan" "*_predict_prior.stan"
    "*_predict_posterior.stan")

foreach(stan_file ${stan_files})
    get_filename_component(root "${stan_file}" NAME_WE)
    
    add_custom_command(
        OUTPUT "${root}.h" "${root}.cpp"
        COMMAND
            ${compile_helper} compile -I ${CMAKE_CURRENT_SOURCE_DIR}
                ${stan_file} "${root}.h" "${root}.cpp"
        DEPENDS ${stan_file})
    list(APPEND header_files "${root}.h")
    list(APPEND source_files "${root}.cpp")
endforeach()

list(SORT header_files)
list(SORT source_files)
list(SORT python_files)
list(SORT stan_files)

pybind11_add_module(_slimp ${source_files} ${header_files})
target_compile_options(_slimp PUBLIC "SHELL:${STAN_CXXFLAGS}")
target_include_directories(_slimp PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_options(_slimp PUBLIC "SHELL:${STAN_LDFLAGS}")
target_link_libraries(_slimp PUBLIC Eigen3::Eigen Python::NumPy ${STAN_LIBS})

install(DIRECTORY DESTINATION "${PYTHON_SITE_PACKAGES}")
install(TARGETS _slimp DESTINATION "${PYTHON_SITE_PACKAGES}/slimp")
install(FILES ${python_files} DESTINATION "${PYTHON_SITE_PACKAGES}/slimp")
