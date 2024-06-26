find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
find_package(SUNDIALS REQUIRED)
find_package(TBB REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE}
        -c "import os; import sysconfig; \
            scheme = {'posix': 'posix_prefix', 'nt': 'nt'}[os.name]; \
            path = sysconfig.get_path('purelib', scheme, {'base': '.'}); \
            print(path.replace(os.path.sep, '/'))"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(
    compile_helper
    ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/slimp/compile.py)

set(STAN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../stan")

file(
    GLOB_RECURSE stan_files
    "${STAN_SOURCE_DIR}/*_sampler.stan"
    "${STAN_SOURCE_DIR}/*_log_likelihood.stan"
    "${STAN_SOURCE_DIR}/*_predict_prior.stan"
    "${STAN_SOURCE_DIR}/*_predict_posterior.stan")
list(SORT stan_files)

file(GLOB_RECURSE header_files "*.h")
file(GLOB_RECURSE source_files "*.cpp")

foreach(stan_file ${stan_files})
    get_filename_component(root "${stan_file}" NAME_WE)
    
    add_custom_command(
        OUTPUT "${root}.h"
        COMMAND
            ${compile_helper} compile -I ${STAN_SOURCE_DIR}
                ${stan_file} "${root}.h"
        DEPENDS ${stan_file})
    list(APPEND header_files "${CMAKE_CURRENT_BINARY_DIR}/${root}.h")
endforeach()

list(SORT header_files)
list(SORT source_files)

file(GLOB_RECURSE python_files "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
list(SORT python_files)

pybind11_add_module(pyslimp ${source_files} ${header_files})

target_compile_options(
    pyslimp PUBLIC -pthread -DSTAN_THREADS -DSTAN_NO_RANGE_CHECKS)
if("${TBB_VERSION}" VERSION_GREATER_EQUAL "2020")
    target_compile_options(pyslimp PUBLIC -DTBB_INTERFACE_NEW)
endif()

target_include_directories(
    pyslimp PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
    "$ENV{CMDSTAN}/stan/src" "$ENV{CMDSTAN}/stan/lib/stan_math")

target_link_libraries(
    pyslimp PUBLIC Boost::boost Eigen3::Eigen Python::NumPy TBB::tbb libslimp)

if("${SUNDIALS_VERSION}" VERSION_GREATER_EQUAL "7.0.0")
    target_link_libraries(libslimp PUBLIC SUNDIALS::core)
else()
    target_link_libraries(libslimp PUBLIC SUNDIALS::generic)
endif()

set_target_properties(
    pyslimp PROPERTIES 
    OUTPUT_NAME _slimp
    $<$<PLATFORM_ID:Darwin>:SUFFIX .so>)

install(DIRECTORY DESTINATION "${PYTHON_SITE_PACKAGES}")
install(TARGETS pyslimp DESTINATION "${PYTHON_SITE_PACKAGES}/slimp")

install(
    DIRECTORY .
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}
    FILES_MATCHING PATTERN *.py)
