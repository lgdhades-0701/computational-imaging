cmake_minimum_required(VERSION 3.17)

include(cmake/PrecompiledHeader.cmake)

project(CudaStereo LANGUAGES CXX)

add_library(imaging STATIC
    imaging/targetver.h
    imaging/stdafx.h
    imaging/stdafx.cpp
    imaging/AlignNaive.h
    imaging/HierarchicalAlignment.h
    imaging/HierarchicalAlignment.cpp
    )

target_include_directories(imaging PUBLIC opencv/build/include)
add_precompiled_header(imaging stdafx.h FORCEINCLUDE SOURCE_CXX imaging/stdafx.cpp)
