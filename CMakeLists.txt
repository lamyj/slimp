cmake_minimum_required(VERSION 3.14)

project("slimp" VERSION 0.3.0)

set(CMAKE_INSTALL_MESSAGE LAZY)

include(CTest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("slimp")
