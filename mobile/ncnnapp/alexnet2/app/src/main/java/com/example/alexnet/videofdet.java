package com.example.alexnet;

import android.annotation.SuppressLint;
import android.content.Context;
import android.net.Uri;
import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;

import com.example.alexnet.ffmpeg.video.videoplayer;

import java.text.SimpleDateFormat;


/**
 * A simple {@link Fragment} subclass.
 * Activities that contain this fragment must implement the
 * {@link videofdet.OnFragmentInteractionListener} interface
 * to handle interaction events.
 * Use the {@link videofdet#newInstance} factory method to
 * create an instance of this fragment.
 */
public class videofdet extends Fragment {
    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER

    private OnFragmentInteractionListener mListener;


    private videoplayer davidPlayer;
    private SurfaceView surfaceView;
    private TextView mTextView,mTextCurTime;
    private SeekBar mSeekBar;
    private boolean isSetProgress=false;
    private static final int HIDE_CONTROL_LAYOUT = -1;

    final static String FVIDEO_BUTTON_MSG="fvideo_button_pushed";

    private String videoPath;

    public videofdet() {
        // Required empty public constructor
    }

    // TODO: Rename and change types and number of parameters
    public static videofdet newInstance(String param1, String param2) {
        videofdet fragment = new videofdet();
        Bundle args = new Bundle();
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
        }
    }


    @SuppressLint("HandlerLeak")
    private Handler handler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if (msg.what == HIDE_CONTROL_LAYOUT) {
                refreshControl();
            } else {
                //  mTextCurTime.setText(formatTime(msg.what));
                mSeekBar.setProgress(msg.what);
            }
            // mSeekBar.setProgress(msg.what);
        }
    };

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view= inflater.inflate(R.layout.fragment_videofdet, container, false);
        surfaceView = view.findViewById(R.id.fsview);
        Button infer_button=(Button)view.findViewById(R.id.fselect_video);
        infer_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                mListener.onVideofdetFragmentInteraction(FVIDEO_BUTTON_MSG);
                stop();
            }
        });

        Button play_button=(Button)view.findViewById(R.id.fvideo_play);
        play_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                player();
            }
        });
        Button pause_button=(Button)view.findViewById(R.id.fvideo_pause);
        pause_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                pause();
            }
        });


        davidPlayer=new videoplayer();
        davidPlayer.setSurfaceView(surfaceView);
        mTextView = view.findViewById(R.id.tview);
        mSeekBar = view.findViewById(R.id.seekBar);
        mTextCurTime = view.findViewById(R.id.tvcur);

        init();
        return view;
    }

    private void init() {
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                //进度改变
                mTextCurTime.setText(formatTime(seekBar.getProgress()));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                //开始拖动
                mTextCurTime.setText(formatTime(seekBar.getProgress()));
                isSetProgress=true;
                refreshControl();
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                //停止拖动
                isSetProgress=false;
                davidPlayer.seekTo(seekBar.getProgress());
                mTextCurTime.setText(formatTime(seekBar.getProgress()));
                refreshControl();
            }
        });
    }

    private String formatTime(long time) {
        SimpleDateFormat format = new SimpleDateFormat("mm:ss");
        return format.format(time);
    }

    public void player() {
        if (videoPath==null||videoPath.isEmpty()){
            Log.d("play video with ffmpeg","invalid video path");
            return;
        }
        davidPlayer.playJava(videoPath);
        if (davidPlayer.getTotalTime() != 0) {
            mTextView.setText(formatTime(davidPlayer.getTotalTime() / 1000));
            mSeekBar.setMax(davidPlayer.getTotalTime() / 1000);
            updateSeekBar();
        }
    }

    public void stop() {
        davidPlayer.stop();
        // Toast.makeText(MainActivity.this,davidPlayer.getTotalTime()+"",Toast.LENGTH_SHORT).show();
        //
        //  mTextView.setText(formatTime(davidPlayer.getTotalTime()/1000));
    }

    public void pause() {
        davidPlayer.pause();
    }



    public void stepback(){
        //快退，seekbar拖动事件
        davidPlayer.stepBack();
    }

    public void stepup(){
        //快进，
        davidPlayer.stepUp();
    }

    //更新进度
    public void updateSeekBar() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true){
                    try {
                        Message message = new Message();
                        double ps= (double)davidPlayer.getCurrentPosition();
                        if(ps==-1){
                            return;
                        }
                        message.what = (int) ps*1000;
                        handler.sendMessage(message);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }).start();
    }

    private void refreshControl() {
        if (isSetProgress) {
            isSetProgress = false;
        } else {
            isSetProgress = true;
            handler.removeMessages(HIDE_CONTROL_LAYOUT);
            handler.sendEmptyMessageDelayed(HIDE_CONTROL_LAYOUT, 1000);
        }
    }


    public void setVideoPath(String path){
        videoPath=path;
    }
    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        if (context instanceof OnFragmentInteractionListener) {
            mListener = (OnFragmentInteractionListener) context;
        } else {
            throw new RuntimeException(context.toString()
                    + " must implement OnFragmentInteractionListener");
        }
    }

    @Override
    public void onDetach() {
        super.onDetach();
        mListener = null;
    }

    /**
     * This interface must be implemented by activities that contain this
     * fragment to allow an interaction in this fragment to be communicated
     * to the activity and potentially other fragments contained in that
     * activity.
     * <p>
     * See the Android Training lesson <a href=
     * "http://developer.android.com/training/basics/fragments/communicating.html"
     * >Communicating with Other Fragments</a> for more information.
     */
    public interface OnFragmentInteractionListener {
        // TODO: Update argument type and name
        void onVideofdetFragmentInteraction(String uri);
    }
}
