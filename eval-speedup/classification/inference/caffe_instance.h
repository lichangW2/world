#ifndef CAFFE_INSTANCE_H 
#define CAFFE_INSTANCE_H

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <ctime>

#include <stdio.h>
#include <sys/time.h>

using namespace caffe;

class CaffeInstance
{
public:
  CaffeInstance(const char* model_file, const char* trained_file){
    caffe_.reset(new Caffe);
    Caffe::Set(caffe_.get());
    Caffe::set_mode(Caffe::GPU);
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    Caffe::Set(NULL);
  }
  shared_ptr<Caffe> getCaffe(){
    return caffe_;
  }

  shared_ptr<Net<float>> getNet(){
    return net_;
  }
private:
    shared_ptr<Caffe> caffe_;
    shared_ptr<Net<float>> net_;
};

#endif // CAFFE_INSTANCE_H
