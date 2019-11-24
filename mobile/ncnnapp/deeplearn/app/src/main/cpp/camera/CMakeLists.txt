# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.
#############################################
cmake_minimum_required(VERSION 3.4.1)


##############################################################
set(OpenCV_DIR /Users/clausewang/Personal/mobile/lib/OpenCV-android-sdk/sdk/native/jni)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV library not found")
endif(OpenCV_FOUND)
##############################################################
## android
if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
    include_directories(${ANDROID_SYSROOT}/usr/include/aarch64-linux-android)
else()
    include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
endif()

# ncnn
#set(NCNN_DIR /Users/clausewang/Personal/mobile/ncnn/build-android-armv7-vulkan/install)
#for os
#set(NCNN_DIR /Users/clausewang/Personal/mobile/ncnn/build/install)
#message("ncnn path: ${NCNN_DIR}")
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION /Users/clausewang/Personal/mobile/lib/ncnn-android-vulkan-lib/${ANDROID_ABI}/libncnn.a)
include_directories(/Users/clausewang/Personal/mobile/lib/ncnn-android-vulkan-lib/include/ncnn)
#include_directories(/Users/clausewang/Personal/mobile/lib/vulkansdk-macos-1.1.92.1/MoltenVK/include/vulkan)
include_directories(../jni ../common .)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(. OUT_SRC_LIST)

add_library( # Sets the name of the library.
        sytletransfer-lib
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${OUT_SRC_LIST})



# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.


find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)


# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.



target_link_libraries( # Specifies the target library.
        sytletransfer-lib
        ncnn
        #/Users/clausewang/Personal/mobile/lib/ncnn-android-vulkan-lib/armeabi-v7a/libncnn.a
        ${OpenCV_LIBS}
        android
        vulkan
        ${log-lib})