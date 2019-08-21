package com.example.alexnet;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

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

    public final static String LOCAL_VIDEO_BUTTON_MSG="video_det";

    private PlayerView mPlayView;

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

        Button infer_button=(Button)view.findViewById(R.id.select_video);
        infer_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                mListener.localVideoInteraction(LOCAL_VIDEO_BUTTON_MSG);
            }
        });

        //========================================

       return view;
    }

    public void setVideoPath(String path){
        if (path==null|| path.isEmpty()){
            return;
        }
        mPlayView.setVideoFilePath(path);
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
