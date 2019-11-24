package com.camera;

import android.Manifest;
import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.TextureView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.nio.ByteBuffer;
import java.util.Arrays;

public class tcamera {

    private static final String TAG = "Camera2Proxy";
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private static final String FRAGMENT_DIALOG = "dialog";
    private boolean isstart=false;

    private Activity mActivity;

    // camera
    private int mCameraId = CameraCharacteristics.LENS_FACING_FRONT; // 要打开的摄像头ID
    private Size mPreviewSize = new Size(640, 480); // 固定640*480演示
    private CameraDevice mCameraDevice; // 相机对象
    private CameraCaptureSession mCaptureSession;

    // handler
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;

    // output
    private Surface mPreviewSurface; // 输出到屏幕的预览
    private ImageReader mImageReader; // 预览回调的接收者
    //private ImageReader.OnImageAvailableListener mOnImageAvailableListener;

    //style tansfer
    private  styletransfer stransfer;

    public void  Init(Activity act, SurfaceView psurface, styletransfer stranf){
        psurface.getHolder().setFixedSize(mPreviewSize.getWidth(),mPreviewSize.getHeight());
        mPreviewSurface=psurface.getHolder().getSurface();

        mActivity=act;
        stransfer=stranf;
    }

    public void SetType(int type){
        if(stransfer!=null){
            stransfer.SetType(type);
        }
    }

    /**
     * 打开摄像头的回调
     */
    private CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            Log.d(TAG, "onOpened");
            mCameraDevice = camera;
            initPreviewRequest();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            Log.d(TAG, "onDisconnected");
            releaseCamera();
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            Log.e(TAG, "Camera Open failed, error: " + error);
            releaseCamera();
        }
    };


    public void openCamera() {
        if(isstart){
            return;
        }
        isstart=true;
        if (ContextCompat.checkSelfPermission(mActivity, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }
        Log.v(TAG, "openCamera");
        startBackgroundThread(); // 对应 releaseCamera() 方法中的 stopBackgroundThread()
        try {
            CameraManager cameraManager = (CameraManager) mActivity.getSystemService(Context.CAMERA_SERVICE);
            Log.d(TAG, "preview size: " + mPreviewSize.getWidth() + "*" + mPreviewSize.getHeight());
            mImageReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(),
                    ImageFormat.YUV_420_888, 2);
            mImageReader.setOnImageAvailableListener(mOnImageAvailableListener, mBackgroundHandler);
            // 打开摄像头
            cameraManager.openCamera(Integer.toString(mCameraId), mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }


    private final ImageReader.OnImageAvailableListener mOnImageAvailableListener
            = new ImageReader.OnImageAvailableListener() {

        @Override
        public void onImageAvailable(ImageReader reader) {

            Image im = reader.acquireNextImage();
            int chcnt = im.getPlanes().length;
            int width = im.getWidth();
            int height = im.getHeight();

            Log.d(TAG, "onImageAvailable " + width + "X" + height + "X" + chcnt);

            ByteBuffer bufferY = im.getPlanes()[0].getBuffer();
            ByteBuffer bufferU = im.getPlanes()[1].getBuffer();
            ByteBuffer bufferV = im.getPlanes()[2].getBuffer();

            ByteBuffer yuvbuffer = ByteBuffer.allocateDirect(bufferY.remaining() + bufferU.remaining() + bufferV.remaining());
            yuvbuffer.put(bufferY);
            yuvbuffer.put(bufferV);
            yuvbuffer.put(bufferU);

            byte[] yuv = yuvbuffer.array();

            stransfer.STransfer(yuv,width,height);

           im.close();
        }
    };

    private void requestCameraPermission() {

        if (mActivity.shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            Log.d(TAG, "requestCameraPermission camera permisson granted");
        } else {
            mActivity.requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    public void releaseCamera() {
        isstart=false;
        Log.v(TAG, "releaseCamera");
        if (mImageReader != null) {
            mImageReader.close();
            mImageReader = null;
        }
        if (mCaptureSession != null) {
            mCaptureSession.close();
            mCaptureSession = null;
        }
        if (mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
        stopBackgroundThread(); // 对应 openCamera() 方法中的 startBackgroundThread()
    }

    /*public void setImageAvailableListener(ImageReader.OnImageAvailableListener onImageAvailableListener) {
        mOnImageAvailableListener = onImageAvailableListener;
    }*/

    public void setPreviewSurface(SurfaceTexture surfaceTexture) {
        // mPreviewSize必须先初始化完成
        surfaceTexture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
        mPreviewSurface = new Surface(surfaceTexture);
    }

    private void initPreviewRequest() {
        try {
            final CaptureRequest.Builder builder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            // 添加输出到屏幕的surface
            builder.addTarget(mPreviewSurface);
            // 添加输出到ImageReader的surface。然后我们就可以从ImageReader中获取预览数据了
            builder.addTarget(mImageReader.getSurface());
            mCameraDevice.createCaptureSession(Arrays.asList(mPreviewSurface, mImageReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            mCaptureSession = session;
                            // 设置连续自动对焦和自动曝光
                            builder.set(CaptureRequest.CONTROL_AF_MODE,
                                    CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                            builder.set(CaptureRequest.CONTROL_AE_MODE,
                                    CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);
                            CaptureRequest captureRequest = builder.build();
                            try {
                                // 一直发送预览请求
                                mCaptureSession.setRepeatingRequest(captureRequest, null, mBackgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            Log.e(TAG, "ConfigureFailed. session: mCaptureSession");
                        }
                    }, mBackgroundHandler); // handle 传入 null 表示使用当前线程的 Looper
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    public Size getPreviewSize() {
        return mPreviewSize;
    }

    public void switchCamera() {
        mCameraId ^= 1;
        Log.d(TAG, "switchCamera: mCameraId: " + mCameraId);
        releaseCamera();
        openCamera();
    }

    private void startBackgroundThread() {
        if (mBackgroundThread == null || mBackgroundHandler == null) {
            Log.v(TAG, "startBackgroundThread");
            mBackgroundThread = new HandlerThread("CameraBackground");
            mBackgroundThread.start();
            mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
        }
    }

    private void stopBackgroundThread() {
        Log.v(TAG, "stopBackgroundThread");
        if(mBackgroundThread==null||mBackgroundHandler==null){
            return;
        }
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}

