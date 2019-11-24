// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/native_window_jni.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <jni.h>

#include <string>
#include <vector>
// ncnn
#include "net.h"

#include "styletransfer.id.h"
#include "styletransfer.param.bin.h"
#include "../common/log.h"

#include <sys/time.h>
#include <unistd.h>

static struct timeval tv_begin;
static struct timeval tv_end;
static double elasped;

ANativeWindow *video_window;

static int REQUIRED_SIZE = 400;

static void bench_start()
{
    gettimeofday(&tv_begin, NULL);
}

static void bench_end(const char* comment)
{
    gettimeofday(&tv_end, NULL);
    elasped = ((tv_end.tv_sec - tv_begin.tv_sec) * 1000000.0f + tv_end.tv_usec - tv_begin.tv_usec) / 1000.0f;
//     fprintf(stderr, "%.2fms   %s\n", elasped, comment);
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "%.2fms   %s", elasped, comment);
}

static int load_net_from_asset(ncnn::Net& net, AAssetManager* mgr, const char* model_path)
{
    // load param
    int ret0 = net.load_param(styletransfer_param_bin);

    // load bin
    AAsset* asset = AAssetManager_open(mgr, model_path, AASSET_MODE_STREAMING);

    off_t start;
    off_t length;
    int fd = AAsset_openFileDescriptor(asset, &start, &length);

    FILE* fp = fdopen(fd, "rb");
    fseek(fp, start, SEEK_CUR);

    int ret1 = net.load_model(fp);

    fclose(fp);// it will close fd too

    AAsset_close(asset);

    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "load_net_from_asset %d %d %d", ret0, ret1, (int)length);

    return 0;
}

static int load_net(ncnn::Net& net, const char* model_path)
{
    // load param
    int ret0 = net.load_param(styletransfer_param_bin);

    FILE* fp = fopen(model_path, "rb");
    int ret1 = net.load_model(fp);

    fclose(fp);// it will close fd too

    __android_log_print(ANDROID_LOG_DEBUG, "load_net", "load_net_from_asset %d %d", ret0, ret1);
    if(ret0!=0||ret1!=0){
        LOGE("load_net failed");
        return 1;
    }

    return 0;
}

void call_image_play(ANativeWindow* window, cv::Mat frame) {
    if (!window||frame.empty()) {
        return;
    }
    LOGE("call_image_play");
    ANativeWindow_setBuffersGeometry(window, frame.cols, frame.rows, WINDOW_FORMAT_RGBA_8888);
    ANativeWindow_Buffer window_buffer;
    if (ANativeWindow_lock(window, &window_buffer, 0)) {
        LOGE("call call_image_play lock failed ");
        return;
    }

    uint8_t *dst = (uint8_t *) window_buffer.bits;
    int dstStride = window_buffer.stride * 4;
    uint8_t *src = frame.data;
    int srcStride = frame.cols*frame.channels();
    LOGE("绘制 宽%d,高%d,stride:%d", window_buffer.width, window_buffer.height,window_buffer.stride);
    for (int i = 0; i < window_buffer.height; ++i) {
        memcpy(dst + i * dstStride, src + i * srcStride, srcStride);
    }
    ANativeWindow_unlockAndPost(window);
    return;
}


static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net styletransfernet[5];

extern "C" {

//在load .so 时系统首先自动调用的方法，用来做一些初始化工作
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}


JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "StyleTransferNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_camera_styletransfer_Init(JNIEnv* env, jobject thiz, jobject surface, jobjectArray jsmodels)
{

    if(surface!= nullptr){
        video_window= ANativeWindow_fromSurface(env, surface);
    }

    if(video_window== nullptr){
        LOGE("StyleTransferNcnn_Init video surface null");
        //initial stage may failed, it would be reset by ResetEnv immediately.
    //   return JNI_FALSE;
    }

    const char* smodels[5];
    jstring smodel_mid[5];

    jsize strArrayLen = env->GetArrayLength(jsmodels);
    if(strArrayLen<=0){
        LOGE("StyleTransferNcnn_Init invalid num of model paths");
        return  JNI_FALSE;
    }
    for (int i=0 ; i<strArrayLen; i++) {
       smodel_mid[i] = (jstring)env->GetObjectArrayElement(jsmodels, i);
       smodels[i] = (char *)env->GetStringUTFChars(smodel_mid[i], 0);
    }
    //====================================================
    LOGI("StyleTransfer gpu num:%d",ncnn::get_gpu_count());

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    //AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    //const char* model_paths[5] = {"candy.bin", "mosaic.bin", "pointilism.bin", "rain_princess.bin", "udnie.bin"};
    for (int i=0; i<strArrayLen; i++)
    {
        styletransfernet[i].opt = opt;

        int ret0 = styletransfernet[i].load_param(styletransfer_param_bin);
        int ret1 = styletransfernet[i].load_model(smodels[i]);
        if (ret0<= 0||ret1!=0){
            LOGE("Load net error, model: %d  param:%d, path:%s",ret0,ret1,smodels[i]);
            return JNI_FALSE;
        }
        LOGE("Load one net, model: %d  param:%d",ret0,ret1);

    }

    for(int i=0;i<strArrayLen;i++){
        env->ReleaseStringUTFChars(smodel_mid[i], smodels[i]);
    }
    env->DeleteLocalRef(jsmodels);
    return JNI_TRUE;
}

// public native Bitmap StyleTransfer(Bitmap bitmap, int style_type, boolean use_gpu);
JNIEXPORT jboolean JNICALL Java_com_camera_styletransfer_StyleTransfer(JNIEnv* env, jobject thiz, jbyteArray frame, jint style, jint width, jint height,jboolean use_gpu)
{
    if (style < 0 || style >= 5){
        LOGI("StyleTransfer invalid type num");
        return JNI_FALSE;
    }

    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0){
         LOGI("StyleTransfer no gpu avilable");
        return JNI_FALSE;
    }


    char *yuv_frame = (char*)env->GetPrimitiveArrayCritical(frame, NULL);
    int size = env->GetArrayLength(frame);
    if(size<10||yuv_frame== nullptr){
        LOGE("detandcompaireandshow failed, invalid image size");
        return JNI_FALSE;
    }



    bench_start();

    LOGI("StyleTransfer frame size:%d, width:%d,height:%d,style:%d",size, width,height,style);
    cv::Mat rgbImg(height, width,CV_8UC3);
    {
        cv::Mat yuvImg;
        yuvImg.create(height * 3/2, width, CV_8UC1);
        int fsize=height *width*3/2;  // only copy part of data
        memcpy(yuvImg.data, yuv_frame, fsize);
        cv::cvtColor(yuvImg, rgbImg, CV_YUV2RGB_NV21);
        yuvImg.release(); //deref 引用数目为0则释放
    }

    LOGI("StyleTransfer yuv to rgb data %x, width:%d,height:%d",rgbImg.data,rgbImg.cols,rgbImg.rows);
    const int downscale_ratio = 2;

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgbImg.data, ncnn::Mat::PIXEL_RGB,width,height, REQUIRED_SIZE, REQUIRED_SIZE);
    LOGI("StyleTransfer ncnn::mat in width:%d,height:%d",in.w,in.h);
    // styletransfer
    ncnn::Mat out;
    {
        ncnn::Extractor ex = styletransfernet[style].create_extractor();
        ex.set_vulkan_compute(use_gpu);

        ex.input(styletransfer_param_id::BLOB_input1, in);

        ex.extract(styletransfer_param_id::BLOB_output1, out);

    }

    if(out.empty()){
        LOGI("StyleTransfer out put empty");
        return JNI_FALSE;
    }

    LOGI("styletransfer out put, channel:%d, width:%d, height:%d",out.c,out.w,out.h);

    try{

        //====================================
        //ncnn to cv::Mat
        cv::Mat out_cv_mat(cv::Size(REQUIRED_SIZE,REQUIRED_SIZE),CV_8UC4);
        out.to_pixels_resize(out_cv_mat.data,ncnn::Mat::PIXEL_RGBA,REQUIRED_SIZE,REQUIRED_SIZE);
        LOGI("styletransfer out put, channel:%d, width:%d, height:%d",out_cv_mat.channels(),out_cv_mat.cols,out_cv_mat.rows);
        call_image_play(video_window,out_cv_mat);
    }catch (...){
        LOGI("StyleTransfer unexception error");
    }


    bench_end("styletransfer");
    env->ReleasePrimitiveArrayCritical(frame, yuv_frame, JNI_ABORT);

    return JNI_TRUE;
}


JNIEXPORT jboolean JNICALL Java_com_camera_styletransfer_ResetSurface(JNIEnv* env, jobject thiz, jobject surface){
    if(surface!= nullptr){
        video_window= ANativeWindow_fromSurface(env, surface);
    }

    if(video_window== nullptr){
        LOGE("ResetSurface video surface nulll");
        //initial stage may failed, it would be reset by ResetEnv immediately.
        return JNI_FALSE;
    }
    LOGE("ResetSurface video surface success");
    return JNI_TRUE;
}

}
