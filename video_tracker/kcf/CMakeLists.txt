cmake_minimum_required(VERSION 2.8)

aux_source_directory(src srcs)
#FILE(GLOB_RECURSE sourcefiles "src/*.cpp")

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/video_tracker/lib)

MESSAGE(STATUS "This is project dir " ${srcs} ${PROJECT_SOURCE_DIR})

add_library(kcf STATIC ${srcs})
set_target_properties(kcf PROPERTIES OUTPUT_NAME "kcf")
set_target_properties(kcf PROPERTIES VERSION 1.0 SOVERSION 1)




