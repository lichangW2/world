package com.example.alexnet;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.content.Intent;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import  android.util.Log;
import android.app.Activity;

import java.io.File;
import java.net.URI;


/**
 * A simple {@link Fragment} subclass.
 * Activities that contain this fragment must implement the
 * {@link ClassifyFragment.OnFragmentInteractionListener} interface
 * to handle interaction events.
 * Use the {@link ClassifyFragment#newInstance} factory method to
 * create an instance of this fragment.
 */
public class ClassifyFragment extends Fragment {
    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
    private static final String ARG_PARAM1 = "param1";
    private static final String ARG_PARAM2 = "param2";

    // TODO: Rename and change types of parameters
    private String mParam1;
    private String mParam2;

    private OnFragmentInteractionListener mListener;
    private ImageView imgv;
    private TextView rettext;

    public ClassifyFragment() {
        // Required empty public constructor
    }

    /**
     * Use this factory method to create a new instance of
     * this fragment using the provided parameters.
     *
     * @param param1 Parameter 1.
     * @param param2 Parameter 2.
     * @return A new instance of fragment BlankFragment.
     */
    // TODO: Rename and change types and number of parameters
    public static ClassifyFragment newInstance(String param1, String param2) {
        ClassifyFragment fragment = new ClassifyFragment();
        Bundle args = new Bundle();
        args.putString(ARG_PARAM1, param1);
        args.putString(ARG_PARAM2, param2);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            mParam1 = getArguments().getString(ARG_PARAM1);
            mParam2 = getArguments().getString(ARG_PARAM2);
        }


        //add by clause
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view= inflater.inflate(R.layout.classify, container, false);

        Button img_button=(Button)view.findViewById(R.id.img_button);
        img_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                onButtonClick(view);
            }
        });

        Button infer_button=(Button)view.findViewById(R.id.infer_button);
        infer_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                mListener.onFragmentInteraction("infer_image");
            }
        });

        imgv=(ImageView)view.findViewById(R.id.classyimage);
        rettext=view.findViewById(R.id.cls_ret);
        return view;
    }

    // TODO: Rename method, update argument and hook method into UI event

    public void onButtonClick(View v) {
        if (mListener != null) {
            mListener.onFragmentInteraction("select_image");
        }
    }

    public void showImageNow(Uri path){
        try{
            Log.d("show iamge","path:"+path);
            //ImageView imgv= this.getView().findViewById(R.id.imageView);
            imgv.setImageURI(path);
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public void showClassifyResult(String result){
        try{
            Log.d("show result","result:"+result);
            //ImageView imgv= this.getView().findViewById(R.id.imageView);
            rettext.setText(result);
        }catch (Exception e){
            e.printStackTrace();
        }
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
     * 与包含这个fragment的通信接口，可以独立定义也可以这样定义在fragment中，党在主activity中定义一个fragment并初始化时会调用onAttach方法传入相关的activity,然后通过这个调用activity实现的方法传递信息；
     * 每个fragment自己handler自己的相关组件的点击，拖拉等各种操作；
     */
    public interface OnFragmentInteractionListener {
        // TODO: Update argument type and name
        void onFragmentInteraction(String uri);
        //String getSelectedImage();
    }
}
