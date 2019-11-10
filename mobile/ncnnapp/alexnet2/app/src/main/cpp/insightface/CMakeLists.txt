# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("OPENMP NOT FOUND !")
endif(OPENMP_FOUND)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
##############################################################
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
##############################################################
## android
if(${ANDROID_ABI} STREQUAL "armeabi-v7a")
    include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
elseif(${ANDROID_ABI} STREQUAL "arm64-v8a")
    include_directories(${ANDROID_SYSROOT}/usr/include/aarch64-linux-android)
else()
    include_directories(${ANDROID_SYSROOT}/usr/include/arm-linux-androideabi)
endif()

file(GLOB iface_files "./insight/*.cpp")
aux_source_directory(. OUT_SRC_LIST)

set(NCNN_SRC_LISTS
        ${iface_files}
        ${OUT_SRC_LIST}
        )

# ncnn
set(NCNN_DIR /Users/clausewang/Personal/mobile/ncnn/build-android-armv7/install)
#for os
#set(NCNN_DIR /Users/clausewang/Personal/mobile/ncnn/build/install)
message("ncnn path: ${NCNN_DIR}")
include_directories(ncnn-android-lib/include)
include_directories(../headers ../jni ./insight ./headers)

add_library( # Sets the name of the library.
        insightface-lib
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${NCNN_SRC_LISTS})



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
        insightface-lib

        ${CMAKE_SOURCE_DIR}/insightface/ncnn-android-lib/${ANDROID_ABI}/libncnn.a
        #-lgomp
        # Links the target library to the log library
        # included in the NDK.
        ${OpenCV_LIBS}
        android
        ${log-lib})
