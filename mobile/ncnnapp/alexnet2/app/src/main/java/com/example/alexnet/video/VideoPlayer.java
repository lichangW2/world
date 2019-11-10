package com.example.alexnet.video;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.widget.TextView;

import com.example.alexnet.face.process.FaceProcess;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class VideoPlayer {

    private static final String TAG = "TAG";
    private static final long TIMEOUT_US = 10000;
    private IPlayerCallBack callBack;
    private VideoThread videoThread;
    private AudioThread audioThread;
    private boolean isPlaying;
    private boolean isPause;
    private String filePath;

    private long sampleDataTime;
    private int count;
    private long duration;

    private FaceProcess faceprocess;

    public VideoPlayer(SurfaceView tg, TextView tg_txt, SurfaceView trace_ret, TextView ret_txt, SurfaceView videoView, String filePath) {
        Log.d("local init", "create videoplayer");
        faceprocess=new FaceProcess();
        faceprocess.InitEnv(tg,trace_ret,videoView,ret_txt);
        this.filePath = filePath;
        isPause=false;
    }

    public void setFilePath(String filePath) {
        if(isPlaying==true||isPause==true){
           stop();
        }
        this.filePath = filePath;
    }

    public void setCallBack(IPlayerCallBack callBack) {
        this.callBack = callBack;
    }

    public boolean isPlaying() {
        return isPlaying;
    }

    public void targetDetect(String path){
        faceprocess.DetAndShow(path);
    }

    public void play() {
        if (filePath==null|| filePath.isEmpty()){
            Log.e("local video start", "please provide video");
            return;
        }
        isPlaying=true;
        if(isPause){
            faceprocess.Pause();
            isPause=false;
            return;
        }
        //if (videoThread == null) {
            videoThread = new VideoThread();

        //}
        //if (audioThread == null) {
            audioThread = new AudioThread();

        //}

        isPause=false;
        sampleDataTime=0;
        count=0;
        duration=0;
        faceprocess.Start();
        videoThread.start();
        audioThread.start();
    }

    public void pause() {
        if(isPlaying==true){
            isPlaying=false;
            isPause = true;
            faceprocess.Pause();
        }
    }
    public void stop() {
        isPause=false;
        isPlaying = false;
        faceprocess.Stop(0);
        destroy();
    }

    public void destroy() {

        if (audioThread != null) audioThread.interrupt();
        if (videoThread != null) videoThread.interrupt();
    }

    public int getCurrentPosition(){
        Log.v(TAG, "count:"+count+" sampledatatime:"+sampleDataTime);
        return (int)(count*sampleDataTime/1000000);
    }

    public int getBufferPercentage(){
        Log.v(TAG, "count:"+count+" sampledatatime:"+sampleDataTime);
         float dur=count*sampleDataTime/1000000;
         float percent=0;
         if (duration!=0){
             percent=dur/dur;
         }
         return (int)percent;
    }
    private void showSupportedColorFormat(MediaCodecInfo.CodecCapabilities caps) {
        System.out.print("supported color format: ");
        for (int c : caps.colorFormats) {
            System.out.print(c + "\t");
        }
        System.out.println();
    }


    /*将缓冲区传递至解码器
     * 如果到了文件末尾，返回true;否则返回false
     */
    private boolean putBufferToCoder(MediaExtractor extractor, MediaCodec decoder, ByteBuffer[] inputBuffers) {
        boolean isMediaEOS = false;
        int inputBufferIndex = decoder.dequeueInputBuffer(TIMEOUT_US);
        if (inputBufferIndex >= 0) {
            ByteBuffer inputBuffer = inputBuffers[inputBufferIndex];
            int sampleSize = extractor.readSampleData(inputBuffer, 0);
            if (sampleSize < 0) {
                decoder.queueInputBuffer(inputBufferIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                isMediaEOS = true;
                Log.v(TAG, "media eos");
            } else {
                decoder.queueInputBuffer(inputBufferIndex, 0, sampleSize, extractor.getSampleTime(), 0);
                extractor.advance();
            }
        }
        return isMediaEOS;
    }

    //获取指定类型媒体文件所在轨道
    private int getMediaTrackIndex(MediaExtractor videoExtractor, String MEDIA_TYPE) {
        int trackIndex = -1;
        for (int i = 0; i < videoExtractor.getTrackCount(); i++) {
            MediaFormat mediaFormat = videoExtractor.getTrackFormat(i);
            String mime = mediaFormat.getString(MediaFormat.KEY_MIME);
            if (mime.startsWith(MEDIA_TYPE)) {
                trackIndex = i;
                break;
            }
        }
        return trackIndex;
    }

    //延迟渲染
    private void sleepRender(MediaCodec.BufferInfo audioBufferInfo, long startMs) {
        while (audioBufferInfo.presentationTimeUs / 1000 > System.currentTimeMillis() - startMs) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                e.printStackTrace();
                break;
            }
        }
    }

    private class VideoThread extends Thread {

        @Override
        public void run() {
            if (faceprocess == null) {
                Log.e("TAG", "faceprocess invalid!");
                return;
            }
            if (filePath.isEmpty()) {
                Log.e("TAG", "video file invalid!");
                return;
            }
            MediaExtractor videoExtractor = new MediaExtractor();
            MediaCodec videoCodec = null;
            try {
                videoExtractor.setDataSource(filePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
            int videoTrackIndex;
            int width=0;
            int height=0;
            //获取视频所在轨道
            videoTrackIndex = getMediaTrackIndex(videoExtractor, "video/");
            if (videoTrackIndex >= 0) {
                MediaFormat mediaFormat = videoExtractor.getTrackFormat(videoTrackIndex);
                 width = mediaFormat.getInteger(MediaFormat.KEY_WIDTH);
                 height = mediaFormat.getInteger(MediaFormat.KEY_HEIGHT);
                duration = mediaFormat.getLong(MediaFormat.KEY_DURATION) / 1000000;
                callBack.videoAspect(width, height, duration);
                videoExtractor.selectTrack(videoTrackIndex);
                try {
                    videoCodec = MediaCodec.createDecoderByType(mediaFormat.getString(MediaFormat.KEY_MIME));
                    showSupportedColorFormat(videoCodec.getCodecInfo().getCapabilitiesForType(mediaFormat.getString(MediaFormat.KEY_MIME)));
                    mediaFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar); //解码为指定的YUV格式，所有设备支持
                    //mediaFormat.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatRGBFlexible);
                    //videoCodec.configure(mediaFormat, surface, null, 0);
                    videoCodec.configure(mediaFormat, null, null, 0);
                    Log.e("TAG", "video mime type:"+mediaFormat.getString(MediaFormat.KEY_MIME));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            if (videoCodec == null) {
                Log.v(TAG, "MediaCodec null");
                return;
            }
            videoCodec.start();

            MediaCodec.BufferInfo videoBufferInfo = new MediaCodec.BufferInfo();
            ByteBuffer[] inputBuffers = videoCodec.getInputBuffers();
            ByteBuffer[] outputBuffers = videoCodec.getOutputBuffers();
            boolean isVideoEOS = false;
            long startMs = System.currentTimeMillis();
            Paint mPaint = new Paint();
            byte[] frame;
            int chcnt=0;
            int image_width=0;
            int image_height=0;


            while (!Thread.interrupted()) {
                if (isPause) {
                    Log.v(TAG, "MediaCodec paused");
                    try{
                        sleep(300);
                    }catch (Exception e){
                        Log.v(TAG, "MediaCodec video sleep failed");
                        e.printStackTrace();
                    }
                    continue;
                }
                if(!isPlaying&&!isPause){
                    break;
                }
                //将资源传递到解码器
                if (!isVideoEOS) {
                    isVideoEOS = putBufferToCoder(videoExtractor, videoCodec, inputBuffers);
                    sampleDataTime= videoExtractor.getSampleTime();
                    if (isVideoEOS){
                        break;
                    }
                }
                int outputBufferIndex = videoCodec.dequeueOutputBuffer(videoBufferInfo, TIMEOUT_US);
                switch (outputBufferIndex) {
                    case MediaCodec.INFO_OUTPUT_FORMAT_CHANGED:
                       // MediaFormat=videoCodec.getOutputFormat();
                        Log.v(TAG, "format changed");
                        break;
                    case MediaCodec.INFO_TRY_AGAIN_LATER:
                        Log.v(TAG, "超时");
                        break;
                    case MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED:
                        outputBuffers = videoCodec.getOutputBuffers();
                        Log.v(TAG, "output buffers changed");
                        break;
                    default:
                        //直接渲染到Surface时使用不到outputBuffer
                        ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];

                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //初始化时直接将surface传给JNI，从c++直接用YUV/RGB 数据渲染surface, 每次从这里给入buffer中的data(byte[]frame)即可;
                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                        //Log.e("TAG", "video mime type:"+frameImage.getFormat()+"width:"+frameImage.getWidth()+"height:"+frameImage.getHeight());
                        //延时操作
                        //如果缓冲区里的可展示时间>当前视频播放的进度，就休眠一下
                        sleepRender(videoBufferInfo, startMs);
                        //渲å染

                       /// if (count==0){
                            Image frameImage=videoCodec.getOutputImage(outputBufferIndex);
                            if(frameImage!=null){
                                //YUV 420_888转RGBA https://stackoverflow.com/questions/30510928/convert-android-camera2-api-yuv-420-888-to-rgb
                                width = frameImage.getWidth();
                                height = frameImage.getHeight();

                                ByteBuffer yBuffer = frameImage.getPlanes()[0].getBuffer();
                                ByteBuffer uBuffer = frameImage.getPlanes()[1].getBuffer();
                                ByteBuffer vBuffer = frameImage.getPlanes()[2].getBuffer();

                                int ySize = yBuffer.remaining();
                                int uSize = uBuffer.remaining();
                                int vSize = vBuffer.remaining();

                                frame=new byte[ySize+uSize+vSize];

                                //U and V are swapped
                                yBuffer.get(frame, 0, ySize);
                                vBuffer.get(frame, ySize, vSize);
                                uBuffer.get(frame, ySize + vSize, uSize);
                                faceprocess.FaceDetAndCompAndDraw(frame,width,height,count);

                                Log.d(TAG, " width:"+ width + "height:" + height +"planes:"+frameImage.getPlanes().length+"format:" + frameImage.getFormat()+"count:"+count+"size:"+(ySize + vSize+uSize));
                                frameImage.close();
                            }

                         //   videoCodec.releaseOutputBuffer(outputBufferIndex, false);
                         //   count++;
                        //    break;
                       // }

                        ///Canvas cavs =surface.lockCanvas(null);

                        ///frame=new byte[videoBufferInfo.size];//BufferInfo内定义了此数据块的大小
                        ///outputBuffer.get(frame);
                        ///outputBuffer.clear();//数据取出后一定记得清空此Buffer MediaCodec是循环使用这些Buffer的
                        ///faceprocess.FaceDetAndCompAndDraw(frame,width,height);
                        ///for (int i=2000;i<55000&&i<frame.length;i++){
                        ///    frame[i]=100;
                        ///}

                        ///YuvImage yuvimage=new YuvImage(frame,ImageFormat.NV21,image_width,image_height,null);//20、20分别是图的宽度与高度

                        ///ByteArrayOutputStream baos = new ByteArrayOutputStream();
                        ///yuvimage.compressToJpeg(new Rect(0, 0,image_width, image_height), 80, baos);//80--JPG图片的质量[0-100],100最高
                        ///byte[] jdata = baos.toByteArray();

                        ///Bitmap btmp=BitmapFactory.decodeByteArray(jdata,0,jdata.length);
                        ///if (cavs==null){
                        ///    Log.v(TAG, "null cavs");
                        ///}
                        ///if (btmp==null){
                        ///    Log.v(TAG, "null btmp,"+videoBufferInfo.size+" lenth:"+frame.length);
                        ///}

                        ///cavs.drawBitmap(btmp,new Rect(0,0,image_width,image_height),new Rect(0,0,cavs.getWidth(),cavs.getHeight()),null);
                        //cavs.drawBitmap(btmp,(float)0.0,(float) 0.0,null); //可以同时写两个不同的视频画面，一大一小
                        ///surface.unlockCanvasAndPost(cavs);
                        videoCodec.releaseOutputBuffer(outputBufferIndex, false);
                        //videoCodec.releaseOutputBuffer(outputBufferIndex, true);
                        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                        count++;
                        break;
                }

                if ((videoBufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    Log.v(TAG, "video buffer stream end");
                    faceprocess.Stop(1);
                    break;
                }
            }//end while
            try {
                faceprocess.Stop(1);
                videoCodec.stop();
                videoCodec.release();
                videoExtractor.release();
                isPlaying=false;
            }catch (Exception e){
                e.printStackTrace();
            }

        }
    }


    private class AudioThread extends Thread {
        private int audioInputBufferSize;

        private AudioTrack audioTrack;

        @Override
        public void run() {
            MediaExtractor audioExtractor = new MediaExtractor();
            MediaCodec audioCodec = null;
            try {
                audioExtractor.setDataSource(filePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
            for (int i = 0; i < audioExtractor.getTrackCount(); i++) {
                MediaFormat mediaFormat = audioExtractor.getTrackFormat(i);
                String mime = mediaFormat.getString(MediaFormat.KEY_MIME);
                if (mime.startsWith("audio/")) {
                    audioExtractor.selectTrack(i);
                    int audioChannels = mediaFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
                    int audioSampleRate = mediaFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    int minBufferSize = AudioTrack.getMinBufferSize(audioSampleRate,
                            (audioChannels == 1 ? AudioFormat.CHANNEL_OUT_MONO : AudioFormat.CHANNEL_OUT_STEREO),
                            AudioFormat.ENCODING_PCM_16BIT);
                    int maxInputSize = mediaFormat.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE);
                    audioInputBufferSize = minBufferSize > 0 ? minBufferSize * 4 : maxInputSize;
                    int frameSizeInBytes = audioChannels * 2;
                    audioInputBufferSize = (audioInputBufferSize / frameSizeInBytes) * frameSizeInBytes;
                    audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,
                            audioSampleRate,
                            (audioChannels == 1 ? AudioFormat.CHANNEL_OUT_MONO : AudioFormat.CHANNEL_OUT_STEREO),
                            AudioFormat.ENCODING_PCM_16BIT,
                            audioInputBufferSize,
                            AudioTrack.MODE_STREAM);
                    audioTrack.play();
                    Log.v(TAG, "audio play");
                    //
                    try {
                        audioCodec = MediaCodec.createDecoderByType(mime);
                        audioCodec.configure(mediaFormat, null, null, 0);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    break;
                }
            }
            if (audioCodec == null) {
                Log.v(TAG, "audio decoder null");
                return;
            }
            audioCodec.start();
            //
            final ByteBuffer[] buffers = audioCodec.getOutputBuffers();
            int sz = buffers[0].capacity();
            if (sz <= 0)
                sz = audioInputBufferSize;
            byte[] mAudioOutTempBuf = new byte[sz];

            MediaCodec.BufferInfo audioBufferInfo = new MediaCodec.BufferInfo();
            ByteBuffer[] inputBuffers = audioCodec.getInputBuffers();
            ByteBuffer[] outputBuffers = audioCodec.getOutputBuffers();
            boolean isAudioEOS = false;
            long startMs = System.currentTimeMillis();

            while (!Thread.interrupted()) {
                if (isPause) {
                    continue;
                }
                if(!isPause&&!isPlaying){
                    break;
                }
                if (!isAudioEOS) {
                    isAudioEOS = putBufferToCoder(audioExtractor, audioCodec, inputBuffers);
                }
                //
                int outputBufferIndex = audioCodec.dequeueOutputBuffer(audioBufferInfo, TIMEOUT_US);
                switch (outputBufferIndex) {
                    case MediaCodec.INFO_OUTPUT_FORMAT_CHANGED:
                        Log.v(TAG, "format changed");
                        break;
                    case MediaCodec.INFO_TRY_AGAIN_LATER:
                        Log.v(TAG, "超时");
                        break;
                    case MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED:
                        outputBuffers = audioCodec.getOutputBuffers();
                        Log.v(TAG, "output buffers changed");
                        break;
                    default:
                        ByteBuffer outputBuffer = outputBuffers[outputBufferIndex];
                        //延时操作
                        //如果缓冲区里的可展示时间>当前视频播放的进度，就休眠一下
                        sleepRender(audioBufferInfo, startMs);
                        if (audioBufferInfo.size > 0) {
                            if (mAudioOutTempBuf.length < audioBufferInfo.size) {
                                mAudioOutTempBuf = new byte[audioBufferInfo.size];
                            }
                            outputBuffer.position(0);
                            outputBuffer.get(mAudioOutTempBuf, 0, audioBufferInfo.size);
                            outputBuffer.clear();
                            if (audioTrack != null)
                                audioTrack.write(mAudioOutTempBuf, 0, audioBufferInfo.size);
                        }
                        //
                        audioCodec.releaseOutputBuffer(outputBufferIndex, false);
                        break;
                }

                if ((audioBufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    Log.v(TAG, "audio buffer stream end");
                    break;
                }
            }//end while
            audioCodec.stop();
            audioCodec.release();
            audioExtractor.release();
            audioTrack.stop();
            audioTrack.release();
        }

    }
}
