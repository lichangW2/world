# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.4.1)


#include opencv
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

##############################################################
## android 
if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
   include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
   include_directories(${ANDROID_SYSROOT}/usr/include/aarch64-linux-android)
else()
   include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
endif()



find_library(log-lib log)
include_directories(../headers ../jni)


# classify-lib.so
aux_source_directory(. CLASSIFICATION_SRC_LIST)
add_library(classify-lib SHARED ${CLASSIFICATION_SRC_LIST})
target_link_libraries(classify-lib ${log-lib} ${OpenCV_LIBS} /Users/clausewang/Personal/mobile/ncnn/build-android-armv7/install/lib/libncnn.a)
