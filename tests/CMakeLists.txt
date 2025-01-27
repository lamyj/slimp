find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

file(GLOB_RECURSE test_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(SORT test_source_files)

foreach(test_source_file ${test_source_files})
    get_filename_component(unit ${test_source_file} NAME_WE)
    
    set(test "${unit}")

    add_executable(test_${test} ${test_source_file})
    target_link_libraries(
        test_${test} PRIVATE libslimp Boost::unit_test_framework)
    set_target_properties(test_${test} PROPERTIES OUTPUT_NAME ${test})
    
    file(READ ${test_source_file} content)
    set(pattern "BOOST_(AUTO|FIXTURE)_TEST_CASE\\(([^),]+)")
    string(REGEX MATCHALL ${pattern} cases ${content})
    
    foreach(case ${cases})
        string(REGEX REPLACE ${pattern} "\\2" case ${case})
        if(NOT ${case} MATCHES "##")
            add_test("${test}_${case}" "${test}" "--run_test=${case}")
        endif()
    endforeach()
endforeach()
