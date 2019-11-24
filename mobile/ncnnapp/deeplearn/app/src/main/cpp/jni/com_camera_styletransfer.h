/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_camera_styletransfer */

#ifndef _Included_com_camera_styletransfer
#define _Included_com_camera_styletransfer
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_camera_styletransfer
 * Method:    stringFromJNI
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_com_camera_styletransfer_stringFromJNI(JNIEnv *, jobject);

JNIEXPORT jboolean JNICALL Java_com_camera_styletransfer_Init(JNIEnv* env, jobject thiz, jobject surface, jobjectArray jarray);

JNIEXPORT jboolean JNICALL Java_com_camera_styletransfer_ResetSurface(JNIEnv* env, jobject thiz, jobject surface);

JNIEXPORT jboolean JNICALL Java_com_camera_styletransfer_StyleTransfer(JNIEnv* env, jobject thiz, jbyteArray frame, jint style, jint width, jint height,jboolean use_gpu);


#ifdef __cplusplus
}
#endif
#endif
