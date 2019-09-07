package com.example.alexnet.ffmpeg.video;

import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class videoplayer implements SurfaceHolder.Callback {
    static{
        System.loadLibrary("avcodec-58");
        System.loadLibrary("avdevice-58");
        System.loadLibrary("avfilter-7");
        System.loadLibrary("avformat-58");
        System.loadLibrary("avutil-56");
        System.loadLibrary("postproc-55");
        System.loadLibrary("swresample-3");
        System.loadLibrary("swscale-5");
        System.loadLibrary("detection-lib");
    }

    private SurfaceView surfaceView;
    public   void playJava(String path) {
        if (surfaceView == null) {
            return;
        }
        play(path);
    }

    public void setSurfaceView(SurfaceView surfaceView) {
        this.surfaceView = surfaceView;
        display(surfaceView.getHolder().getSurface());
        surfaceView.getHolder().addCallback(this);
    }

    public native void play(String path);

    public native void display(Surface surface);

    public native void  stop();

    public native void pause();

    public native int getTotalTime();

    public native double getCurrentPosition();

    public native void seekTo(int msec);


    public native void stepBack();

    public native void stepUp();


    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {

    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {
        display(surfaceHolder.getSurface());
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {

    }
}
