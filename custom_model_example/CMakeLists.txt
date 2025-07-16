cmake_minimum_required(VERSION 3.14)

project("my_model")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
# NOTE: see remarks abount SUNDIALS below
# find_package(SUNDIALS REQUIRED)
find_package(TBB REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import slimp; print(slimp.__path__[0])"
    OUTPUT_VARIABLE SLIMP_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
set(compile_helper "${Python_EXECUTABLE}" "${SLIMP_PATH}/compile.py")

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/my_model.h"
    COMMAND ${compile_helper} compile 
        "${CMAKE_CURRENT_SOURCE_DIR}/my_model.stan"
        "${CMAKE_CURRENT_BINARY_DIR}/my_model.h"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/my_model.stan")

pybind11_add_module(
    my_model
    "${CMAKE_CURRENT_SOURCE_DIR}/my_model.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/my_model.h")

target_compile_options(my_model PUBLIC -pthread -DSTAN_THREADS)
if("${TBB_VERSION}" VERSION_GREATER_EQUAL "2020")
    target_compile_options(my_model PUBLIC -DTBB_INTERFACE_NEW)
endif()

target_include_directories(
    my_model PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
    "$ENV{CMDSTAN}/stan/src" "$ENV{CMDSTAN}/stan/lib/stan_math"
    # WARNING: due to splitting of C and C++ header files in SUNDIALS 7,
    # only SUNDIALS 6 is usable: use headers of version bundled with cmdstan
    "$ENV{CMDSTAN}/stan/lib/stan_math/lib/sundials_6.1.1/include")

target_link_libraries(
    my_model PUBLIC Boost::boost Eigen3::Eigen Python::NumPy TBB::tbb slimp)
