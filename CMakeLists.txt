cmake_minimum_required(VERSION 3.14)

project("slimp" VERSION 1.0.0)

option(BUILD_SHARED_LIBS "Build with shared libraries." ON)

set(CMAKE_INSTALL_MESSAGE LAZY)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("src/lib")
add_subdirectory("src/python")
add_subdirectory("tests")
