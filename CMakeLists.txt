cmake_minimum_required(VERSION 3.17)

option(PACKAGE_TESTS "Build the tests" ON)

include(cmake/PrecompiledHeader.cmake)
add_subdirectory("extern/gflags")
# TODO(fyhuang): hack for MDd
target_compile_options(gflags_nothreads_static PUBLIC "/MDd")

project(CudaStereo LANGUAGES CXX)

set(OPENCV_ROOT "${PROJECT_SOURCE_DIR}/extern/opencv/build")
add_library(opencv SHARED IMPORTED GLOBAL)
set_property(TARGET opencv PROPERTY IMPORTED_IMPLIB_RELEASE "${OPENCV_ROOT}/x64/vc15/lib/opencv_world420.lib")
set_property(TARGET opencv PROPERTY IMPORTED_LOCATION_RELEASE "${OPENCV_ROOT}/x64/vc15/bin/opencv_world420.dll")
set_property(TARGET opencv PROPERTY IMPORTED_IMPLIB_DEBUG "${OPENCV_ROOT}/x64/vc15/lib/opencv_world420d.lib")
set_property(TARGET opencv PROPERTY IMPORTED_LOCATION_DEBUG "${OPENCV_ROOT}/x64/vc15/bin/opencv_world420d.dll")

add_library(imaging STATIC
    imaging/targetver.h
    imaging/stdafx.h
    imaging/stdafx.cpp
    imaging/Util.h
    imaging/BlockAligner.h
    imaging/AlignNaive.h
    imaging/HierarchicalAlignment.h
    imaging/HierarchicalAlignment.cpp
)

target_include_directories(imaging PUBLIC extern/opencv/build/include)
target_link_libraries(imaging PUBLIC opencv)
add_precompiled_header(imaging stdafx.h FORCEINCLUDE SOURCE_CXX imaging/stdafx.cpp)
# From this SO post: <https://stackoverflow.com/a/59594095>
add_custom_command(TARGET imaging POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:opencv> $<TARGET_FILE_DIR:imaging>
)

# TODO(fyhuang): use target_compile_options
if (MSVC)
    # warning level 4 and all warnings as errors
    add_compile_options(/W4 /WX)
else()
    # lots of warnings and all warnings as errors
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()

# Test application
add_executable(align_image
    align_image/main.cpp
)
target_link_libraries(align_image PRIVATE gflags::gflags imaging)
target_include_directories(align_image PRIVATE "${PROJECT_SOURCE_DIR}")

if(PACKAGE_TESTS)
    enable_testing()
    include(GoogleTest)

    add_subdirectory("extern/googletest")
    mark_as_advanced(
        BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
        gmock_build_tests gtest_build_samples gtest_build_tests
        gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
    )

    add_executable(imaging_tests
        imaging/tests/main.cpp
        imaging/tests/AlignNaiveTest.cpp
        imaging/tests/HierarchicalAlignmentTest.cpp
    )
    target_link_libraries(imaging_tests gtest gmock gtest_main imaging)
    target_include_directories(imaging_tests PUBLIC "${PROJECT_SOURCE_DIR}")
    gtest_discover_tests(imaging_tests
        WORKING_DIRECTORY "${PROJECT_DIR}"
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
                   VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${PROJECT_DIR}/extern/opencv/build/x64/vc15/bin"
    )
endif()
