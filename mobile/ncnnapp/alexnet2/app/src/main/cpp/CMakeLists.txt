# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

#include opencvå
set(OpenCV_DIR /Users/clausewang/Personal/mobile/lib/OpenCV-android-sdk/sdk/native/jni)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else(OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif(OpenCV_FOUND)

#include ncnn
#include ncnn lib
#for android
set(NCNN_DIR /Users/clausewang/Personal/mobile/ncnn/build-android-armv7/install)
#for os
#set(NCNN_DIR /Users/clausewang/Personal/mobile/ncnn/build/install)
message("ncnn path: ${NCNN_DIR}")
include_directories(${NCNN_DIR}/include)
#for android
find_library(NCNN_LIBS ncnn NAMES libncnn.a PATHS /Users/clausewang/Personal/mobile/ncnn/build-android-armv7/install/)
#for macos
#find_library(NCNN_LIBS ncnn NAMES libncnn.a PATHS /Users/clausewang/Personal/mobile/ncnn/build/install/)


aux_source_directory(. SRC_LIST)
include_directories(headers jni)

add_library(native-lib SHARED ${SRC_LIST})

find_library(log-lib log)


target_link_libraries(native-lib ${log-lib} ${OpenCV_LIBS} /Users/clausewang/Personal/mobile/ncnn/build-android-armv7/install/lib/libncnn.a)
