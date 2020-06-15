cmake_minimum_required(VERSION 3.17)

option(PACKAGE_TESTS "Build the tests" ON)

include(extern/win32cmake/All.cmake)
include(extern/win32cmake/PrecompiledHeader.cmake)

add_subdirectory("extern/gflags")
force_target_msvc_crt(gflags_nothreads_static)

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE "extern/eigen-3.3.7")

project(ComputationalImaging LANGUAGES CXX)

set(OpenCV_DIR "extern/opencv")
find_package(OpenCV)

# Imaging library
add_library(imaging STATIC
    imaging/targetver.h
    imaging/stdafx.h
    imaging/stdafx.cpp
    imaging/Util.h
    imaging/SuperResPipeline.h
    imaging/alignment/block_aligner.h
    imaging/alignment/block_align.h
    imaging/alignment/block_align.cpp
    imaging/alignment/lucas_kanade.h
    imaging/alignment/lucas_kanade.cpp
    #imaging/alignment/hier_align.h
    #imaging/alignment/hier_align.cpp
    imaging/alignment/robustness_mask.h
    imaging/rendering/kernel_reconstruction.h
)

get_target_property(imaging_sources imaging SOURCES)
source_group(TREE "${PROJECT_SOURCE_DIR}/imaging" FILES ${imaging_sources})

target_include_directories(imaging PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(imaging PUBLIC opencv_world eigen)
add_precompiled_header(imaging stdafx.h FORCEINCLUDE SOURCE_CXX imaging/stdafx.cpp)
target_post_build_copy_dll(imaging opencv_world)
target_wall_werror(imaging)
target_compile_options(imaging PRIVATE /wd4505)

# gentestimg library
add_library(gentestimg STATIC
    gentestimg/image_generator.h
    gentestimg/image_generator.cpp
)

target_include_directories(gentestimg PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(gentestimg PUBLIC opencv_world)
target_wall_werror(gentestimg)

add_executable(gentestimg_cmd
    cmd/gentestimg.cpp
)
target_link_libraries(gentestimg_cmd PRIVATE gflags::gflags gentestimg)
set_target_properties(gentestimg_cmd PROPERTIES OUTPUT_NAME gentestimg)


# Test application
#add_executable(align_image
#    align_image/main.cpp
#)
#target_link_libraries(align_image PRIVATE gflags::gflags imaging)
#target_include_directories(align_image PRIVATE "${PROJECT_SOURCE_DIR}")

# Super resolution app
#add_executable(super_res
#    super_res/main.cpp
#)
#target_link_libraries(super_res PRIVATE gflags::gflags imaging)
#target_include_directories(super_res PRIVATE "${PROJECT_SOURCE_DIR}")

if(PACKAGE_TESTS)
    enable_testing_gtest("extern/googletest")
    add_gtest_tests(imaging_tests
        imaging/tests/main.cpp
        imaging/tests/test_util.cpp
        imaging/tests/alignment/test_block_align.cpp
        imaging/tests/alignment/test_lucas_kanade.cpp
        #imaging/tests/alignment/test_hier_align.cpp
    )
    target_link_libraries(imaging_tests imaging)
endif()
