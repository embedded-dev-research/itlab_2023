

file(GLOB_RECURSE TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM TEST_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/model_read/model_read.cpp")


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


target_include_directories(run_test PRIVATE "${CMAKE_SOURCE_DIR}/app/ReaderImage")

if (WIN32)
    add_custom_command(TARGET run_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}/Debug/."
            "${CMAKE_BINARY_DIR}/bin/")
    add_custom_command(TARGET run_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/3rdparty/opencv/build/bin/${CMAKE_BUILD_TYPE}/Release/."
            "${CMAKE_BINARY_DIR}/bin/")
    if(TENSORFLOW_FOUND)
        add_custom_command(TARGET run_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                "${CMAKE_SOURCE_DIR}/3rdparty/tensorflow/lib/tensorflow.dll"
                "${CMAKE_BINARY_DIR}/bin/")
    endif()
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

file(DOWNLOAD
    "https://tinyurl.com/2w6eh53v"
    "${CMAKE_CURRENT_BINARY_DIR}/squeeze.tflite"
    SHOW_PROGRESS
    STATUS status_code
    LOG log_file
)

target_compile_definitions(run_test PUBLIC TESTS_BINARY_PATH="${CMAKE_CURRENT_BINARY_DIR}")

