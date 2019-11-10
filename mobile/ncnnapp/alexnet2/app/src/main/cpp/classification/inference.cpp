//
// Created by clausewang(王立昌) on 2019-06-02.
//

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "net.h"
#include "mat.h"

#include "sort.h"
#include "inference.h"
#include "utils.h"

using namespace std;

Env::Env(const char* model, const char* param, const char* label, float *mean, int input_size) {

    this->net=new(ncnn::Net);
    int load_p_su=this->net->load_param(param);
    LOGI("load classify param:%d",load_p_su);
    //int load_b_su=this->net->load_model(model);
    //LOGI("load classify  model:%d",load_b_su);
    this->labels=loadLabels(label);
    LOGI("load classify label:%d",this->labels.size());
    this->input_size=input_size;

    this->mean[0]=mean[0];
    this->mean[1]=mean[1];
    this->mean[2]=mean[2];
}


Env::~Env() {
    this->net->clear();
    delete this->net;
}


std::string Env::Inference(const char* image,int limit){
    try {
        cv::Mat cvin = cv::imread(image, cv::IMREAD_COLOR);
        if(cvin.empty()||cvin.cols<10||cvin.rows<10){
            return "invalid input image";
        }
        cout<<"cvmat1:"<<cvin.size<<endl;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(cvin.data, ncnn::Mat::PIXEL_BGR, cvin.cols, cvin.rows,
                                                     input_size, input_size);

        cout<<"cvmat:"<<cvin.size<<"ncnnmat:"<<in.dims<<" "<<in.c<<endl;
        //inference
        vector<pair<std::size_t,float > > out;
        run(in, mean, out);

        stringstream sout;
        cout<<"inference out:"<<endl;
        for (int i=0;i<limit&i<out.size();i++){
            sout<<" { index:"<<out[i].first<<" label:"<<labels[out[i].first]<<" value:"<<out[i].second<<"} ";
        }
        sout<<endl;
        return sout.str();
    }catch (std::exception& e){
        cout<<"catch exception:"<<e.what()<<endl;
    }
}


int Env::run(ncnn::Mat& in, const float mean[],vector<pair<std::size_t,float > >& out){

    ncnn::Mat o_out;
    in.substract_mean_normalize(mean,0);
    ncnn::Extractor extra=net->create_extractor();
    extra.input("data",in);
    extra.extract("prob",o_out);

    cout<<"out_dim:"<<o_out.dims<<"c:"<<o_out.c<<"h:"<<o_out.h<<"w:"<<o_out.w<<endl;

    ncnn::Mat o_out_flatterned=o_out.reshape(o_out.c*o_out.h*o_out.w);
    vector<float> fout;
    fout.assign((float*)o_out_flatterned.data,(float*)o_out_flatterned.data+o_out.c*o_out.h*o_out.w);
    out=sort_indexs(fout);
    return 0;
}

vector<string> Env::loadLabels(const char* label_file ){

    vector<string> labels;
    fstream fin(label_file);
    string readline;

    while(getline(fin,readline)){
        if(readline.empty()){
            cerr<<"empty line in label file !!!"<<endl;
            break;
        }
        labels.push_back(readline);
    }
    return labels;
}

