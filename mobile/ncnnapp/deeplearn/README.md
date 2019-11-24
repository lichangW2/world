## MOBILE android armv7
=======================================================

**Tec Stack**

```
  android+NDK+JNI+C_PLUS_PLUS+NCNN+VULKAN+(CV+NLP+OCR)
```

**NOTE**

```
1. Thread 异步加载模型，不影响UI渲染
2. MediaCodec硬解码比FFMPEG软解码效率高，但是解码的YUV数据在android java中处理很低效率，因此直接通过JNI 处理MediaCodec和Camera（摄像头）得到的YUV数据，并在C++中通过JNI直接渲染到SurfaceView进行播放；
3. MediaCodec直接可用, FFMPEG需要compile for android
4. 利用 Android 提供的 AudioRecord 采集音频，利用 AudioTrack 播放音频，利用 MediaCodec 来编解码，这些 API 均是 Android 提供的 Java 层 API，无论是采集、播放还是编解码，这些 API 接口都需要将音频数据从 Java 拷贝到 native 层，或者从 native 层拷贝到 Java, OpenSL ES 全称是：Open Sound Library for Embedded Systems，是一套无授权费、跨平台、针对嵌入式系统精心优化的硬件音频加速API.
5. CMake中的shared lib都要显式打包进apk并System.loadLibrary，因为shared lib是运行时链接.
6. FFPEG可以直接定位到某个时间戳附近的关键帧然后继续播放，可以以此做音视频的快进快退和跳转.
7. adb logcat 抽取移动端详细log，stacktrace.
8. MediaCodic物理解码速度很快，但是face det因算法不同可能一帧处理超过40ms导致视频卡顿，因此采用多进程，多缓冲方式可以解决问题；即3进程检测，2进程对比，1进程显示；845四大核四小核不能浪费^_^;
```

**ISSUE**

```
1. javah 找不到android类
   用classpath指定所有类的位置，路径分隔符:或;(window) 
   javah -v -jni -classpath /Users/clausewang/Library/Android/sdk/platforms/android-29/android.jar:.  -d /Users/clausewang/Personal/mobile/ncnnapp/alexnet2/app/src/main/cpp/jni com.example.alexnet.ffmpeg.video.videoplayer

2. mediacodec解码的视频帧在c++中转为openv YUV cv::Mat时只需要memcpy 3/2的height*width数据，如果copy全部的原始YUV数据cv::Mat会越界;

3. 新建ndk c++ native工程之后需要在project structure中修改ndk为对应的新版本r16或r17等
```
