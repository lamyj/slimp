find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
find_package(SUNDIALS REQUIRED)
find_package(TBB REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)

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
    "${STAN_SOURCE_DIR}/*_predict_posterior.stan"
    
    "${STAN_SOURCE_DIR}/*/sampler.stan"
    "${STAN_SOURCE_DIR}/*/log_likelihood.stan"
    "${STAN_SOURCE_DIR}/*/predict_prior.stan"
    "${STAN_SOURCE_DIR}/*/predict_posterior.stan")
list(SORT stan_files)

file(GLOB_RECURSE header_files "*.h")
file(GLOB_RECURSE source_files "*.cpp")

foreach(stan_file ${stan_files})
    file(RELATIVE_PATH relative_stan_path "${STAN_SOURCE_DIR}" "${stan_file}")
    get_filename_component(relative_stan_dir "${relative_stan_path}" DIRECTORY)
    get_filename_component(name "${relative_stan_path}" NAME_WE)
    set(header "${CMAKE_CURRENT_BINARY_DIR}/${relative_stan_dir}/${name}.h")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${relative_stan_dir}")
    if(NOT "${relative_stan_dir}" STREQUAL "")
        add_custom_command(
            OUTPUT "${header}"
            COMMAND
                ${compile_helper} compile -I ${STAN_SOURCE_DIR}
                    -p "${relative_stan_dir}_"
                    ${stan_file} "${header}"
            DEPENDS ${stan_file})
    else()
        add_custom_command(
            OUTPUT "${header}"
            COMMAND
                ${compile_helper} compile -I ${STAN_SOURCE_DIR}
                    ${stan_file} "${header}"
            DEPENDS ${stan_file})
    endif()
    list(APPEND header_files "${header}")
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
    pyslimp PUBLIC Boost::boost Eigen3::Eigen Python::NumPy TBB::tbb
    xtensor xtensor-python libslimp)

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
