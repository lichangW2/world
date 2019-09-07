#include <jni.h>
#include <string>

#include "net.h"
#include "mat.h"

#include "sort.h"
#include "inference.h"
#include "com_example_alexnet_netlib.h"
#include "utils.h"


extern "C" JNIEXPORT jstring JNICALL
Java_com_example_alexnet_netlib_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    LOGI("this is native c++ %d",5);
    std::string hello = "Hello from clause & C++";

    return env->NewStringUTF(hello.c_str());
}


extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_netlib_initEnv
        (JNIEnv * jenv, jobject jobj, jstring jmodel, jstring jparam, jstring jlabel, jfloatArray jmean, jint jmean_size, jint jinput_size){

       const char* cmodel = jenv->GetStringUTFChars(jmodel,JNI_FALSE);
       const char* cparam = jenv->GetStringUTFChars(jparam,JNI_FALSE);
       const char* clabel = jenv->GetStringUTFChars(jlabel,JNI_FALSE);

       jfloat* cmean=jenv->GetFloatArrayElements(jmean,JNI_FALSE);

       Env* _infer_env=new Env(cmodel,cparam,clabel,cmean,jinput_size);

       jenv->ReleaseFloatArrayElements(jmean,cmean,0);
       jenv->ReleaseStringUTFChars(jmodel,cmodel);
       jenv->DeleteLocalRef(jmodel);
       jenv->ReleaseStringUTFChars(jparam,cparam);
       jenv->DeleteLocalRef(jparam);
       jenv->ReleaseStringUTFChars(jlabel,clabel);
       jenv->DeleteLocalRef(jlabel);
    LOGI("init finished env_ptr:%d",_infer_env);
    return (jlong)_infer_env;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_example_alexnet_netlib_inference
        (JNIEnv *jenv, jobject jobj, jlong _infer_env, jstring jimage, jint jlimit){

    const char* cimage = jenv->GetStringUTFChars(jimage,JNI_FALSE);

    std::string clret=((Env*)_infer_env)->Inference(cimage,jlimit);

    jenv->ReleaseStringUTFChars(jimage,cimage);
    jenv->DeleteLocalRef(jimage);
    return jenv->NewStringUTF(clret.c_str());
}

//del env function