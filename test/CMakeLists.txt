file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if (NOT WIN32)
    find_package(OpenMP REQUIRED)
endif()

add_executable(run_test ${TEST_SRC_FILES})
if (NOT WIN32)
    target_link_libraries(run_test PUBLIC OpenMP::OpenMP_CXX)
endif()
target_link_libraries(run_test PUBLIC perf_lib layers_lib)
target_link_libraries(run_test PUBLIC gtest_main)
target_link_libraries(run_test PUBLIC ReadLib)
target_link_libraries(run_test PUBLIC reader_lib)


target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

if (WIN32)
	if ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	    set(CMAKE_BUILD_TYPE "Debug")
	endif()
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
	    set(CMAKE_BUILD_TYPE "Release")
	endif()
    add_custom_command(TARGET run_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}/."
            "${CMAKE_BINARY_DIR}/bin/")
endif()

add_test(UnitTests ${CMAKE_BINARY_DIR}/bin/run_test)

file(DOWNLOAD
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"
    "${CMAKE_CURRENT_BINARY_DIR}/image.jpg"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

file(DOWNLOAD
    "https://raw.githubusercontent.com/bpinaya/AlexNetRT/master/data/alexnet/imagenet-labels.txt"
    "${CMAKE_CURRENT_BINARY_DIR}/imagenet-labels.txt"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)
target_compile_definitions(run_test PRIVATE TEST_DATA_PATH="${CMAKE_SOURCE_DIR}/test/model_read/json_for_test")


target_compile_definitions(run_test PUBLIC TESTS_BINARY_PATH="${CMAKE_CURRENT_BINARY_DIR}")

