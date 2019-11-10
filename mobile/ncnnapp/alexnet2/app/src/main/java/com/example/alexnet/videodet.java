package com.example.alexnet;

import android.content.Context;
import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import com.example.alexnet.face.process.FaceProcess;
import com.example.alexnet.video.PlayerView;


/**
 * A simple {@link Fragment} subclass.
 * Activities that contain this fragment must implement the
 * {@link videodet.OnFragmentInteractionListener} interface
 * to handle interaction events.
 */
public class videodet extends Fragment {
    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER

    // TODO: Rename and change types of parameters
    private OnFragmentInteractionListener mListener;

    public final static String LOCAL_VIDEO_BUTTON_MSG="face_video_det";
    public final static String LOCAL_IMAGE_BUTTON_MSG="face_image_det";

    private PlayerView mPlayView;
    private String target_path;

    public videodet() {
        // Required empty public constructor
    }


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
       View view= inflater.inflate(R.layout.local_detection, container, false);
        mPlayView= view.findViewById(R.id.show_video);

        Button video_select_button=(Button)view.findViewById(R.id.select_face_video);
        video_select_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                mPlayView.stop();
                mListener.localVideoInteraction(LOCAL_VIDEO_BUTTON_MSG);
            }
        });
        Button infer_button=(Button)view.findViewById(R.id.select_face_target);
        infer_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                mListener.localVideoInteraction(LOCAL_IMAGE_BUTTON_MSG);
            }
        });

        Button target_show_button=(Button)view.findViewById(R.id.show_face_target);
        target_show_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                if(target_path!=null&&!target_path.isEmpty()){
                    mPlayView.TargetDetect(target_path);
                }
            }
        });


        SurfaceView target=view.findViewById(R.id.det_target);
        SurfaceView trace_ret=view.findViewById(R.id.det_ret);
        TextView target_text=view.findViewById(R.id.det_target_text);
        TextView ret_text=view.findViewById(R.id.det_ret_text);
        Log.d("videodet","onCreateView start");
        mPlayView.init(target,target_text,trace_ret,ret_text);

        Log.d("videodet","onCreateView end");
        //========================================

       return view;
    }

    public void setVideoPath(String path){
        if (path==null|| path.isEmpty()){
            return;
        }
        mPlayView.setVideoFilePath(path);
    }

    public void targetDetect(String path){
        //image path
        if (path==null|| path.isEmpty()){
            return;
        }
        target_path=path;
    }

    @Override
    public void onPause() {
        super.onPause();
        mPlayView.pause();
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
        void localVideoInteraction(String message);
    }
}
