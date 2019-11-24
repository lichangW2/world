package com.camera;

import android.graphics.Bitmap;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.content.res.AssetManager;
import android.view.SurfaceView;

public class styletransfer implements SurfaceHolder.Callback{
    static {
        System.loadLibrary("sytletransfer-lib");
    }

    private int stype=0;
    private SurfaceView tvideo;

    public void SetType(int sp){
        stype=sp;
    }

    public void InitEnv( SurfaceView video,String[] models){
        if(tvideo!=null){
            Init(tvideo.getHolder().getSurface(),models);
        }else{
            Init(null,models);
        }

        SetVideoSurface(video);
    }
    public void SetVideoSurface(SurfaceView video){
        tvideo=video;
        if (tvideo!=null){
            tvideo.getHolder().addCallback(this);
            ResetSurface(tvideo.getHolder().getSurface());
        }
    }
    public void STransfer(byte[] image, int width, int height){
        StyleTransfer(image, stype, width,height,true);
    }

    public native boolean Init( Surface video,String[] models);
    public native boolean ResetSurface(Surface video);
    public native boolean StyleTransfer(byte[] image, int type,int width, int height, boolean gpu);
    public native String stringFromJNI();

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {
        Log.d("styletransfer","surfaceChanged");
        ResetSurface(surfaceHolder.getSurface());
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {

    }
}
