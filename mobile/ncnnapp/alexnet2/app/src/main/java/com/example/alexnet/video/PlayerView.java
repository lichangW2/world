package com.example.alexnet.video;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.widget.MediaController;

public class PlayerView extends SurfaceView implements MediaController.MediaPlayerControl,IPlayerCallBack {

    private double aspectRatio;
    private MediaController mMediaController;
    private VideoPlayer mVideoPlayer;

    private long duration;

    public PlayerView(Context context) {
        this(context, null);
    }

    public PlayerView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public PlayerView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        mVideoPlayer = new VideoPlayer(this.getHolder().getSurface());
        mVideoPlayer.setCallBack(this);
        mMediaController = new MediaController(getContext());
        mMediaController.setMediaPlayer(this);
    }

    private void attachMediaController() {
        View anchorView = this.getParent() instanceof View ? (View) this.getParent() : this;
        mMediaController.setAnchorView(anchorView);
        mMediaController.setEnabled(true);
    }


    public MediaController getMediaController() {
        return mMediaController;
    }

    public void setVideoFilePath(String videoFilePath) {
        mVideoPlayer.setFilePath(videoFilePath);
    }

    public void setAspect(double aspect) {
        if (aspect > 0) {
            this.aspectRatio = aspect;
            //don't reset layout
           // requestLayout();
        }
    }

   /*
   don't reset layout
   @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        if (aspectRatio > 0) {
            int initialWidth = MeasureSpec.getSize(widthMeasureSpec);
            int initialHeight = MeasureSpec.getSize(heightMeasureSpec);

            final int horizPadding = getPaddingLeft() + getPaddingRight();
            final int vertPadding = getPaddingTop() + getPaddingBottom();
            initialWidth -= horizPadding;
            initialHeight -= vertPadding;

            final double viewAspectRatio = (double) initialWidth / initialHeight;
            final double aspectDiff = aspectRatio / viewAspectRatio - 1;

            if (Math.abs(aspectDiff) > 0.01) {
                if (aspectDiff > 0) {
                    initialHeight = (int) (initialWidth / aspectRatio);
                } else {
                    initialWidth = (int) (initialHeight * aspectRatio);
                }
                initialWidth += horizPadding;
                initialHeight += vertPadding;
                widthMeasureSpec = MeasureSpec.makeMeasureSpec(initialWidth, MeasureSpec.EXACTLY);
                heightMeasureSpec = MeasureSpec.makeMeasureSpec(initialHeight, MeasureSpec.EXACTLY);
            }
        }
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
    }
*/
    @Override
    public void start() {
        Log.e("playview", "start" );
        mVideoPlayer.play();
    }

    @Override
    public void pause() {
        Log.e("playview", "pause" );
        mVideoPlayer.pause();
    }

    @Override
    public int getDuration() {
        Log.e("playview", "duration:"+duration );
        return (int)duration;
    }

    @Override
    public int getCurrentPosition() {
        Log.e("playview", "getcurrentposition" );
        mVideoPlayer.getCurrentPosition();
        return 0;
    }

    @Override
    public void seekTo(int pos) {
        Log.e("playview", "seekTo" );
        mVideoPlayer.stop();
    }

    @Override
    public boolean isPlaying() {
        Log.e("playview", "isplaying" );
        return mVideoPlayer.isPlaying();
    }

    @Override
    public int getBufferPercentage() {
        Log.e("playview", "getBufferPercentage" );
        mVideoPlayer.getBufferPercentage();
        return 0;
    }

    @Override
    public boolean canPause() {
        return true;
    }

    @Override
    public boolean canSeekBackward() {
        return true;
    }

    @Override
    public boolean canSeekForward() {
        return true;
    }

    @Override
    public int getAudioSessionId() {
        return 1;
    }

    @Override
    public void videoAspect(final int width, final int height, final long time) {
        post(new Runnable() {
            @Override
            public void run() {
                duration=time;
                setAspect((float) width / height);
            }
        });
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        attachMediaController();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        Log.e("playview", "onTouchEvent" );
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (mMediaController != null)
                if (!mMediaController.isShowing()) {
                    mMediaController.show();
                } else {
                    mMediaController.hide();
                }
        }
        return super.onTouchEvent(event);
    }
}
