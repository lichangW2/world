# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

cmake_minimum_required(VERSION 3.4.1)

#ffmpeg
set(distribution_DIR /Users/clausewang/Personal/mobile/lib/ffmpeg-4.0.2/android/arm/lib)
add_library( avcodec-58
             SHARED
             IMPORTED)
set_target_properties( avcodec-58
                       PROPERTIES IMPORTED_LOCATION
                       ${distribution_DIR}/libavcodec-58.so)
add_library( avfilter-7
             SHARED
             IMPORTED)
set_target_properties( avfilter-7
                       PROPERTIES IMPORTED_LOCATION
                       ${distribution_DIR}/libavfilter-7.so)

add_library( avformat-58
             SHARED
             IMPORTED)
set_target_properties( avformat-58
                       PROPERTIES IMPORTED_LOCATION
                       ${distribution_DIR}/libavformat-58.so)

add_library( avutil-56
             SHARED
             IMPORTED)
set_target_properties( avutil-56
                       PROPERTIES IMPORTED_LOCATION
                       ${distribution_DIR}/libavutil-56.so)

add_library( swresample-3
             SHARED
             IMPORTED)
set_target_properties( swresample-3
                       PROPERTIES IMPORTED_LOCATION
                       ${distribution_DIR}/libswresample-3.so)

add_library( swscale-5
             SHARED
             IMPORTED)
set_target_properties( swscale-5
                       PROPERTIES IMPORTED_LOCATION
                       ${distribution_DIR}/libswscale-5.so)

include_directories(/Users/clausewang/Personal/mobile/lib/ffmpeg-4.0.2/android/arm/include)

if(${ANDROID_ABI} STREQUAL "armeabi-v7a") 
   include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi) 
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a") 
   include_directories(${ANDROID_SYSROOT}/usr/include/aarch64-linux-android) 
else() 
   include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi) 
endif()

############################################################
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

############################################################
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

############################################################
find_library(log-lib log)
include_directories(../detection_headers ../headers ../jni)


# classify-lib.so
aux_source_directory(. CLASSIFICATION_SRC_LIST)
add_library(detection-lib SHARED ${CLASSIFICATION_SRC_LIST})
target_link_libraries(detection-lib 
avcodec-58
avfilter-7
avformat-58
avutil-56
swresample-3
swscale-5
android
OpenSLES
${log-lib} 
${OpenCV_LIBS} 
/Users/clausewang/Personal/mobile/ncnn/build-android-armv7/install/lib/libncnn.a
)
