/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_example_alexnet_ffmpeg_video_videoplayer */

#ifndef _Included_com_example_alexnet_ffmpeg_video_videoplayer
#define _Included_com_example_alexnet_ffmpeg_video_videoplayer
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    play
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_play
  (JNIEnv *, jobject, jstring);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    display
 * Signature: (Landroid/view/Surface;)V
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_display
  (JNIEnv *, jobject, jobject);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    release
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_stop
  (JNIEnv *, jobject);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    stop
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_pause
  (JNIEnv *, jobject);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    getTotalTime
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_getTotalTime
  (JNIEnv *, jobject);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    getCurrentPosition
 * Signature: ()D
 */
JNIEXPORT jdouble JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_getCurrentPosition
  (JNIEnv *, jobject);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    seekTo
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_seekTo
  (JNIEnv *, jobject, jint);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    stepBack
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_stepBack
  (JNIEnv *, jobject);

/*
 * Class:     com_example_alexnet_ffmpeg_video_videoplayer
 * Method:    stepUp
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_example_alexnet_ffmpeg_video_videoplayer_stepUp
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
