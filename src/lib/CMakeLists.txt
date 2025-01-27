find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
find_package(SUNDIALS REQUIRED)
find_package(TBB REQUIRED)
find_package(xtensor REQUIRED)

file(GLOB_RECURSE header_files "*.h")
file(GLOB_RECURSE source_files "*.cpp")
file(GLOB_RECURSE template_files "*.txx")

list(SORT header_files)
list(SORT source_files)
list(SORT template_files)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

add_library(libslimp ${source_files} ${header_files})

target_compile_definitions(
    libslimp PRIVATE $<$<PLATFORM_ID:Windows>:BUILDING_SLIMP>)
        
target_compile_options(
    libslimp PUBLIC -pthread -DSTAN_THREADS -DSTAN_NO_RANGE_CHECKS)
if("${TBB_VERSION}" VERSION_GREATER_EQUAL "2020")
    target_compile_options(libslimp PUBLIC -DTBB_INTERFACE_NEW)
endif()

target_include_directories(
    libslimp PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/> $<INSTALL_INTERFACE:>
        "$ENV{CMDSTAN}/stan/src" "$ENV{CMDSTAN}/stan/lib/stan_math")

target_link_libraries(
    libslimp PUBLIC
    Boost::boost Eigen3::Eigen Python::NumPy Python::Python TBB::tbb xtensor)

if("${SUNDIALS_VERSION}" VERSION_GREATER_EQUAL "7.0.0")
    target_link_libraries(libslimp PUBLIC SUNDIALS::core)
else()
    target_link_libraries(libslimp PUBLIC SUNDIALS::generic)
endif()

set_target_properties(
    libslimp PROPERTIES OUTPUT_NAME slimp VERSION ${slimp_VERSION})

install(
    TARGETS libslimp
    EXPORT slimpTargets
    ARCHIVE DESTINATION bin LIBRARY DESTINATION lib RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/slimp
    INCLUDES DESTINATION include)

foreach(include_file ${header_files} ${template_files})
    file(
        RELATIVE_PATH 
        include_path ${CMAKE_CURRENT_SOURCE_DIR} ${include_file})
    get_filename_component(include_path ${include_path} PATH)
    install(FILES ${include_file} DESTINATION "include/${include_path}")
endforeach()
