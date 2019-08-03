#include <jni.h>
#include <string>

#include "net.h"
#include "mat.h"
#include "sort.h"
#include "inference.h"
#include "jni/com_example_alexnet_MainActivity.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_alexnet_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}


extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_MainActivity_initEnv
        (JNIEnv * env, jobject jobj, jstring model, jstring param, jstring label, jfloatArray mean, jint mean_size, jint input_size){

       const char* cmodel = env->GetStringUTFChars(model,JNI_FALSE);
       const char* cparam = env->GetStringUTFChars(param,JNI_FALSE);
       const char* clabel = env->GetStringUTFChars(label,JNI_FALSE);

       jfloat* cmean=env->GetFloatArrayElements(mean,JNI_FALSE);


       env->ReleaseStringUTFChars(model,cmodel);
       env->DeleteLocalRef(model);
       env->ReleaseStringUTFChars(param,cparam);
       env->DeleteLocalRef(param);
       env->ReleaseStringUTFChars(label,clabel);
       env->DeleteLocalRef(label);
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_alexnet_MainActivity_inference
        (JNIEnv *, jobject, jlong, jstring, jint){

}