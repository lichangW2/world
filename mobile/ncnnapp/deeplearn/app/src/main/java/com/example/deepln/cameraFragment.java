package com.example.deepln;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Spinner;

import androidx.fragment.app.Fragment;

import com.camera.styletransfer;
import com.camera.tcamera;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class cameraFragment extends Fragment {

    private tcamera stcamera;
    private SurfaceView transfer_view;
    private SurfaceView snapshop_view;
    private boolean isinit=false;

    private OnFragmentInteractionListener mListener;

    private Spinner sspinner;
    private ArrayAdapter<String> adapter;
    private List<String> list = new ArrayList<String>();

    public cameraFragment(){
        stcamera=new tcamera();
    }


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view= inflater.inflate(R.layout.styletransfer, container, false);

        Button start_button=(Button)view.findViewById(R.id.camera_start);
        start_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                if(!isinit) {
                    if(mListener.gotTransfer()==null){return;}
                    styletransfer stranf=mListener.gotTransfer();
                    stranf.SetVideoSurface(transfer_view);
                    stcamera.Init(getActivity(),snapshop_view,stranf);
                    isinit=true;
                }
                if (stcamera!=null){
                    stcamera.openCamera();
                }
            }
        });

        Button stop_button=(Button)view.findViewById(R.id.camera_stop);
        stop_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                if (stcamera!=null){
                    stcamera.releaseCamera();
                }
            }
        });

        sspinner=(Spinner)view.findViewById(R.id.style_spinner);
        transfer_view=(SurfaceView)view.findViewById(R.id.camera_surface);
        snapshop_view=(SurfaceView)view.findViewById(R.id.camer_snapshot);

        if(mListener.gotTransfer()!=null){
            styletransfer stranf=mListener.gotTransfer();
            stranf.SetVideoSurface(transfer_view);
            stcamera.Init(getActivity(),snapshop_view,stranf);
            isinit=true;
        }
        //====================================================================
        //spinner
        list.add("风格1");
        list.add("风格2");
        //list.add("风格3");
        //list.add("风格4");
        //list.add("风格5");
        adapter = new ArrayAdapter<String>(getActivity(), android.R.layout.simple_spinner_item, list);
        //第三步：设置下拉列表下拉时的菜单样式
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        //第四步：将适配器添加到下拉列表上
        sspinner.setAdapter(adapter);
        //第五步：添加监听器，为下拉列表设置事件的响应
        sspinner.setOnItemSelectedListener(new Spinner.OnItemSelectedListener(){
            public void onItemSelected(AdapterView<?> argO, View argl, int arg2, long arg3) {
                // TODO Auto-generated method stub
                /* 将所选spinnertext的值带入myTextView中*/
                Log.v("cameraFragment", "onResume");
                /* 将 spinnertext 显示^*/
                String select=argO.getItemAtPosition(arg2).toString();

                switch (select){
                    case "风格1":
                        stcamera.SetType(0);
                        Log.v("cameraFragment spinner", "select:"+select);
                        break;
                    case  "风格2":
                        stcamera.SetType(1);
                        Log.v("cameraFragment spinner", "select:"+select);
                        break;
                    case  "风格3":
                        stcamera.SetType(2);
                        Log.v("cameraFragment spinner", "select:"+select);
                        break;
                    case  "风格4":
                        stcamera.SetType(3);
                        Log.v("cameraFragment spinner", "select:"+select);
                        break;
                    case  "风格5":
                        stcamera.SetType(4);
                        Log.v("cameraFragment spinner", "select:"+select);
                        break;
                    default:
                        Log.v("cameraFragment spinner", "select:"+select);
                        break;
                }

                argO.setVisibility(View.VISIBLE);
            }
            public void onNothingSelected(AdapterView<?> argO) {
                // TODO Auto-generated method stub

                argO.setVisibility(View.VISIBLE);
            }
        });


        return view;
    }

    @Override
    public void onResume() {
        Log.v("cameraFragment", "onResume");
        super.onResume();
        if (transfer_view.isActivated()) {
            stcamera.openCamera();
        } else {
           // mCameraView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    public void onPause() {
        Log.v("cameraFragment", "onPause");
        super.onPause();
        stcamera.releaseCamera();
    }

    public void onHide() {
        if (stcamera!=null){
            Log.v("cameraFragment", "onHide");
            stcamera.releaseCamera();
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

    public interface OnFragmentInteractionListener {
        // TODO: Update argument type and name
        String[] gotModelPath();
        public styletransfer gotTransfer();
        //String getSelectedImage();
    }

}
