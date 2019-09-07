//
// Created by clausewang(王立昌) on 2019-06-02.
//

#ifndef ALEXNET_INFERENCE_H
#define ALEXNET_INFERENCE_H

#include <string>
#include <vector>

#include "net.h"

class Env final{

public:
    Env()=delete;
    Env(const char* model, const  char* param,const char* label, float *mean, int input_size);
    std::string Inference(const char* img, int limit);
    ~Env();

private:
    std::vector<std::string> loadLabels(const char* label_file );
    int run(ncnn::Mat& in, const float mean[],std::vector<std::pair<size_t,float > >& out);

private:
    ncnn::Net* net;
    std::vector<std::string> labels;
    float mean[3];
    int input_size;

};

#endif //ALEXNET_INFERENCE_H
