cmake_minimum_required(VERSION 2.8)

project(serving_eval CXX)

include_directories(include)
add_library(ipcv STATIC IPConv.cpp)
#add_executable(iv IPConv.cpp)

add_subdirectory(test)
enable_testing()
add_test(NAME ipcv_test COMMAND Test)