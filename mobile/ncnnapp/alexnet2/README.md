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

```
