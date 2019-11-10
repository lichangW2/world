package com.example.alexnet.face.process;

import android.graphics.PixelFormat;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.TextView;

public class FaceProcess implements SurfaceHolder.Callback{
    //det detect and compair
    static {
        System.loadLibrary("insightface-lib");
    }

    private final  String LOG_TAG="faceprocess";
    private long env;

    private String target_name;
    private String trace_name;
    private String video_name;

    private SurfaceView target_sv;
    private SurfaceView trace_sv;
    private SurfaceView video_sv;

    private TextView trace_text;

    public void InitEnv(SurfaceView tg, SurfaceView ret, SurfaceView video,TextView traceText) {
        if (tg==null||ret==null||video==null){
            Log.d(LOG_TAG,"init face processor failed");
            return;
        }

        Log.d(LOG_TAG,"initEnv before");

        trace_text=traceText;
        //must be assigned to a local variable, or it would not work in jni
        target_sv=tg;
        trace_sv=ret;
        video_sv=video;
        env=initEnv(target_sv.getHolder().getSurface(),trace_sv.getHolder().getSurface(),video_sv.getHolder().getSurface());

        target_sv.getHolder().addCallback(this);
        trace_sv.getHolder().addCallback(this);
        video_sv.getHolder().addCallback(this);

        target_name=target_sv.getHolder().toString();
        trace_name=trace_sv.getHolder().toString();
        video_name=video_sv.getHolder().toString();

        Log.d(LOG_TAG,"initEnv,taget_holder:"+target_sv.getHolder());
        if(env==0){
            Log.d(LOG_TAG,"init face processor failed");
        }
    }
    public void Pause(){
        if (env!=0){
            pause(env);
        }
    }
    public  void Stop(int end){
        if (env!=0){
            Log.d(LOG_TAG,"videoplayer stop");
            stop(env,end);
        }
    }

    public  void Start(){
        if (env!=0){
            start(env);
        }
    }

    public  void FaceDetAndCompAndDraw(byte[] frame,int width, int height,int offset){
        String trace_t=detAndCompairAndShow(env,frame, width, height,offset);
        trace_text.setText(trace_t);
    }
    public  void DetAndShow(String image_path){
        detAndShow(env,image_path);
    }



    public native long initEnv(Surface tg, Surface ret, Surface video);
    public native long resetEnv(long env,Surface tg, Surface ret, Surface video);
    public native long start(long env);
    public native long pause(long env);
    public native long stop(long env,int end);
    public native String detAndShow(long env, String img_path);
    public native String detAndCompairAndShow(long env, byte[] frame,int width, int height,int offset);


    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {

       // Log.d(LOG_TAG,"surface changed:"+surfaceHolder.toString()+"target_name:"+target_name+"trace_name:"+trace_name+"video_name:"+video_name);
        if(surfaceHolder.toString().equals(target_name)){
            env=resetEnv(env,surfaceHolder.getSurface(),null,null);
        }
        if(surfaceHolder.toString().equals(trace_name)){
            env=resetEnv(env,null,surfaceHolder.getSurface(),null);
        }
        if(surfaceHolder.toString().equals(video_name)){
            env=resetEnv(env,null,null,surfaceHolder.getSurface());
        }

        if(env==-1){
            Log.d(LOG_TAG,"surface changed, name:"+surfaceHolder.toString()+"failed .....");
        }

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {

    }

}
