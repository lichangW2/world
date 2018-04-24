cmake_minimum_required(VERSION 3.9)
project(objtrack)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${custome_libs})
add_executable(objtrack main.cpp)
target_link_libraries(objtrack ${OpenCV_LIBS})