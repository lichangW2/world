/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
#include <string>
#include <android/native_window_jni.h>

#include "face_pro.h"
#include "utils.h"
#include "com_example_alexnet_face_process_FaceProcess.h"
/*
 * Class:     com_example_alexnet_face_process_FaceProcess
 * Method:    initEnv
 * Signature: (Landroid/view/Surface;Landroid/view/Surface;Landroid/view/Surface;)J
 */

extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_face_process_FaceProcess_initEnv
  (JNIEnv * env, jobject self, jobject target, jobject trace, jobject video){

    if(target== nullptr){
        LOGE("init faceprocess env failed target nulll");
        return (jlong)-1;
    }
    if(trace== nullptr){
        LOGE("init faceprocess env failed trace nulll");
        return (jlong)-1;
    }
    if(video== nullptr){
        LOGE("init faceprocess env failed video nulll");
       //return (jlong)-1;
    }
    ANativeWindow * target_surface = ANativeWindow_fromSurface(env, target);
    ANativeWindow * trace_surface = ANativeWindow_fromSurface(env, trace);
    ANativeWindow * video_surface = ANativeWindow_fromSurface(env, video);

    if(target_surface== nullptr){
        LOGE("init faceprocess env failed target surface nulll");
        //initial stage may failed, it would be reset by ResetEnv immediately.
        //return (jlong)-1;
    }
    if(trace_surface== nullptr){
        //initial stage may failed, it would be reset by ResetEnv immediately.
        LOGE("init faceprocess env failed trace surface nulll");
        //return (jlong)-1;
    }
    if(video_surface== nullptr){
        LOGE("init faceprocess env failed video surface nulll");
        //initial stage may failed, it would be reset by ResetEnv immediately.
        //return (jlong)-1;
    }
    FaceEnv* fenv=new FaceEnv(target_surface, nullptr, nullptr);

   return (jlong)(fenv);
}

extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_face_process_FaceProcess_resetEnv
        (JNIEnv * env, jobject self, jlong fenv,jobject target, jobject trace, jobject video ){

    //lock
    FaceEnv* fv=(FaceEnv*)fenv;
    if(fv== nullptr){
        LOGE("reset env failed, null env");
        return jlong (-1);
    }
    if(target!= nullptr){
        LOGI("reset env target surface");
        ANativeWindow * target_surface = ANativeWindow_fromSurface(env, target);
        if(target_surface== nullptr){
            LOGE("reset env failed target surface nulll");
            return (jlong)-1;
        }
        fv->ResetEnv(target_surface, nullptr, nullptr);
        return (jlong)(fv);
    }
    if(trace!= nullptr){
        LOGI("reset env trace surface");
        ANativeWindow * trace_surface = ANativeWindow_fromSurface(env, trace);
        if(trace_surface== nullptr){
            LOGE("reset env failed trace surface nulll");
            return (jlong)-1;
        }
        fv->ResetEnv(nullptr, trace_surface, nullptr);
        return (jlong)(fv);
    }
    if(video!= nullptr){
        LOGI("reset env video surface");
        ANativeWindow * video_surface = ANativeWindow_fromSurface(env, video);
        if(video_surface== nullptr){
            LOGE("reset env failed video surface nulll");
            return (jlong)-1;
        }
        fv->ResetEnv(nullptr, nullptr, video_surface);
        return (jlong)(fv);
    }
}

extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_face_process_FaceProcess_start
        (JNIEnv *env, jobject self, jlong fenv){
    LOGI("FaceProcess_start ...");
    FaceEnv* fv=(FaceEnv*)fenv;
    fv->Start();
    return (jlong)(fv);
}

/*
 * Class:     com_example_alexnet_face_process_FaceProcess
 * Method:    pause
 * Signature: (J)J
 */

extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_face_process_FaceProcess_pause
  (JNIEnv *env, jobject self, jlong fenv){
    LOGI("FaceProcess_pause ...");
    FaceEnv* fv=(FaceEnv*)fenv;
    fv->Pause();
    return (jlong)(fv);
}

/*
 * Class:     com_example_alexnet_face_process_FaceProcess
 * Method:    stop
 * Signature: (J)J
 */
extern "C" JNIEXPORT jlong JNICALL Java_com_example_alexnet_face_process_FaceProcess_stop
  (JNIEnv *env, jobject self, jlong fenv,jint end){
    LOGI("FaceProcess_stop ...");
    FaceEnv* fv=(FaceEnv*)fenv;
    fv->isend=end;
    fv->Stop();
    return (jlong)(fv);
}

/*
 * Class:     com_example_alexnet_face_process_FaceProcess
 * Method:    detAndShow
 * Signature: (JLandroid/view/SurfaceView;Ljava/lang/String;)Ljava/lang/String;
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_example_alexnet_face_process_FaceProcess_detAndShow
  (JNIEnv *env, jobject self, jlong fenv, jstring jimage){
    LOGE("FaceProcess_detAndShow");
    const char* cimage = env->GetStringUTFChars(jimage,JNI_FALSE);
    FaceEnv* fv=(FaceEnv*)fenv;
    char* ret=fv->DetAndShow(cimage);

    LOGE("FaceProcess_detAndShow end");
    env->ReleaseStringUTFChars(jimage,cimage);
    env->DeleteLocalRef(jimage);
    std::string hello = "success";
    return env->NewStringUTF(hello.c_str());
}

/*
 * Class:     com_example_alexnet_face_process_FaceProcess
 * Method:    detAndCompairAndShow
 * Signature: (J[B)Ljava/lang/String;
 */
extern "C" JNIEXPORT jstring JNICALL Java_com_example_alexnet_face_process_FaceProcess_detAndCompairAndShow
  (JNIEnv * env, jobject self, jlong fenv, jbyteArray frame,jint width,jint height, jint offset){

    LOGI("detandcompaireandshow width:%d, height:%d,env:%X",width,height,fenv);
    FaceEnv* fv=(FaceEnv*)fenv;
    if(fv== nullptr){
        LOGE("detandcompaireandshow env failed, null env");
        return env->NewStringUTF("null env");
    }
    char *yuv_frame = (char*)env->GetPrimitiveArrayCritical(frame, NULL);
    int size = env->GetArrayLength(frame);
    if(size<10){
        LOGE("detandcompaireandshow failed, invalid image size");
        return env->NewStringUTF("null env");
    }
    char *yuv = (char *)malloc(size);
    memcpy(yuv, yuv_frame, size);

    FrameInfo f=FrameInfo{
            frame:yuv,
            size:size,
            width:(int)width,
            height:(int)height,
            offset:(int)offset
    };
    fv->PutFrame(f);
    env->ReleasePrimitiveArrayCritical(frame, yuv_frame, JNI_ABORT);
    return env->NewStringUTF(fv->trace_text);
}

