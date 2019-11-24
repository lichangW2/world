//
// Created by clausewang(王立昌) on 2019-11-12.
//

#include <jni.h>
#include <string>
#include <../jni/com_camera_styletransfer.h>
#include <../common/log.h>
extern "C" JNIEXPORT jstring JNICALL Java_com_camera_styletransfer_stringFromJNI
(JNIEnv * env, jobject) {

    LOGI("this is native c++ %d",5);
    std::string hello = "Hello from clause & C++";
    return env->NewStringUTF(hello.c_str());
}