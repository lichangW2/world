cmake_minimum_required(VERSION 3.9)
project(objtrack)

set(CMAKE_CXX_STANDARD 11)

SET(custome_headers "/Users/cj/workspace/world/video_tracker/kcf/src")
SET(custome_libs "/Users/cj/workspace/world/video_tracker/lib")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS} ${custome_headers})

message(STATUS "headers" ${custome_headers} " ##source_dir has changed " ${PROJECT_SOURCE_DIR})

link_directories(${custome_libs})
add_executable(objtrack main.cpp)
target_link_libraries(objtrack ${OpenCV_LIBS} libkcf.a)