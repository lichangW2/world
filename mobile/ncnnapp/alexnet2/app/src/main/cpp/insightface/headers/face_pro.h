//
// Created by clausewang(王立昌) on 2019-10-20.
//

#ifndef CPP_FACE_PRO_H
#define CPP_FACE_PRO_H

#include <android/log.h>
#include <android/native_window_jni.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "../insight/interface.h"

#include <vector>
#include <queue>


#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, "jni", __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO , "jni", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN , "libssd", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "libssd", __VA_ARGS__))

#define FACE_DETECT_SIZEH   448


struct FrameInfo {
    char * frame;

    int size;
    int width;
    int height;
    int offset;
};

struct CompFaceInfo{
    cv::Mat in;
    std::vector<FaceInfo> faces;

    int width;
    int height;
    int offset;
};


class FaceEnv{

public:
    FaceEnv(ANativeWindow* target_surface,ANativeWindow* trace_surface, ANativeWindow* video_surface);
    void ResetEnv(ANativeWindow* target_surface,ANativeWindow* trace_surface, ANativeWindow* video_surface);
    char* DetAndShow(const char* img_path);
    //int VideoDetAndShow(char* yuv, int size,int width,int height, int );
    int Start();
    int Pause();
    int Stop();

   ~FaceEnv();

public:
    void PutFrame(FrameInfo f);
    FrameInfo GetFrame();

    void PutShowFrame(cv::Mat f);
    cv::Mat GetShowFrame();

    void PutCompareFace(CompFaceInfo f);
    CompFaceInfo GetCompareFace();

    ANativeWindow *GetVideoWindow(){return video_window;}
    ANativeWindow *GetTraceWindow(){return trace_window;}
    ANativeWindow *GetTargetWindow(){return target_window;}


    std::vector<float> GetTargetFaceFeature(){return target_face_feature;}

    int isplaying;  //0: not playing, 1:playing;
    int ispause;
    int isend;
    int isdetend;

    char* trace_text;
    float compare_threshold;
    pthread_mutex_t fdmutex;

private:

    int faceCurrentNum;

    cv::Mat target_face_img;
    std::vector<float> target_face_feature;


    pthread_mutex_t smutex;
    pthread_cond_t scond;
    //同步锁
    pthread_mutex_t fmutex;
    //条件变量
    pthread_cond_t fcond;

    //同步锁
    pthread_mutex_t cmutex;
    //条件变量
    pthread_cond_t ccond;

    pthread_t spid;
    pthread_t cpid;
    std::vector<pthread_t*> fpids;//处理线程

    //原图队列
    std::queue<FrameInfo> fqueue;
    //显示队列
    std::queue<cv::Mat> squeue;
    //人脸队列
    std::queue<CompFaceInfo> cqueue;

    ANativeWindow *target_window = 0;
    ANativeWindow *trace_window = 0;
    ANativeWindow *video_window = 0;
};




#endif //CPP_FACE_PRO_H
