#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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
#include "queue.h"
#include "caffe_instance.h"

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

class Classifier {
 public:
  Classifier(const char* name, const char* model_file, const char* trained_file, int instance_num);
  ~Classifier(){}; //todo

  std::vector<float> Classify(const char* buffer, size_t length);
 private:

  void InitImage(const cv::Mat& img, cv::Mat* out_img);
  void CopyImage(const cv::Mat& img, std::vector<cv::Mat>* input_channels, shared_ptr<Net<float> > net); 


 private:
  Queue<std::unique_ptr<CaffeInstance>> instance_pool; 

  cv::Size input_geometry_;
  int num_channels_;
  int instance_num_;
  string name_;
};


Classifier::Classifier(const char* name, const char* model_file_in, const char* trained_file_in, int instance_num) {

  string model_file(model_file_in); 
  string trained_file(trained_file_in);
  name_ = name;
  instance_num_ = instance_num;
  std::cout << name_ << "  instance_num_ : " << instance_num_ << std::endl;

  for (int i = 0; i < instance_num_; ++i)
  {
     std::unique_ptr<CaffeInstance>  instance;
     instance.reset( new CaffeInstance(model_file_in, trained_file_in));
     instance_pool.Push(std::move(instance));
  }

  std::unique_ptr<CaffeInstance> instance = instance_pool.Pop();
  Caffe::Set(instance->getCaffe().get());
  shared_ptr<Net<float> > net = instance->getNet();
  Caffe::set_mode(Caffe::GPU);

  Blob<float>* input_layer = net->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 )
    << "Input layer should have 3 channels."; 
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
  Blob<float>* output_layer = net->output_blobs()[0];

  instance_xpool.Push(std::move(instance));
  Caffe::Set(NULL);

}

std::vector<float> Classifier::Classify(const char* buffer, size_t length){

  clock_t start_time1,end_time1,end_time2, end_time3;
  start_time1 = clock();

  // 1. read image
  //string file(buffer);
  //cv::Mat img = cv::imread(file, -1);
  cv::_InputArray array(buffer, length);
  cv::Mat img = cv::imdecode(array, -1);

  cv::Mat shaped_img;
  InitImage(img, &shaped_img);

  clock_t end_time1_1 = clock();

  // 2. get instance & reshape 
  std::unique_ptr<CaffeInstance> instance = instance_pool.Pop();
  Caffe::Set(instance->getCaffe().get());
  shared_ptr<Net<float> >net = instance->getNet();

  Blob<float>* input_layer = net->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net->Reshape();

  // 3. copy image
  std::vector<cv::Mat> input_channels;
  CopyImage(shaped_img, &input_channels, net);
  end_time1 = clock();

  //4. forward
  net->Forward();
  end_time2 = clock();

  //5. copy out
  Blob<float>* output_layer = net->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  end_time3 = clock();

  // 6. putback instance
  instance_pool.Push(std::move(instance));
  Caffe::Set(NULL);

  double seconds1_1 = (double)(end_time1_1-start_time1)/CLOCKS_PER_SEC;
  double seconds1_2 = (double)(end_time1-end_time1_1)/CLOCKS_PER_SEC;
  double seconds2 = (double)(end_time2-end_time1)/CLOCKS_PER_SEC;
  double seconds3 = (double)(end_time3-end_time2)/CLOCKS_PER_SEC;
  std::cout<< name_ << "\tpre_1:" << seconds1_1<< " pre_2:"<<seconds1_2<<"s forward:"<<seconds2<<"s after:"<< seconds3<<"s"<< std::endl;

  return std::vector<float>(begin, end);
}

void Classifier::InitImage(const cv::Mat& img,
                            cv::Mat* out_img) {

  int width = input_geometry_.width;
  int height = input_geometry_.height;

  cv::Mat sample;
  if (img.channels() == 4 )
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 )
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  sample_resized.convertTo(*out_img, CV_32FC3);

}

void Classifier::CopyImage(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels, shared_ptr<Net<float> > net) {

  int width = input_geometry_.width;
  int height = input_geometry_.height;

  Blob<float>* input_layer = net->input_blobs()[0];
  float* input_data = input_layer->mutable_cpu_data();

  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }

  cv::split(img, *input_channels);
}

struct classifier_ctx
{
    Classifier* classifier;
};

int isGoogleLoggingInitialized = 0;
extern "C" classifier_ctx* classifier_initialize(const char* name, const char* model_file, const char* trained_file,
                                     int instance_num)
{
  try
  {
    if (!isGoogleLoggingInitialized){
      isGoogleLoggingInitialized = 1;
      ::google::InitGoogleLogging("inference_server");
    }
    Classifier*classifier =  new Classifier(name, model_file, trained_file, instance_num);
    classifier_ctx* ctx = new classifier_ctx;
    ctx->classifier = classifier;
    errno = 0;
    return ctx;
  }
  catch (const std::invalid_argument& ex)
  {
    LOG(ERROR) << "exception: " << ex.what();
    errno = EINVAL;
    return NULL;
  }
}

//todo: buffer now as local file name .to change as image data
extern "C" const char* classifier_classify(classifier_ctx* ctx,
                                const char* buffer, size_t length)
{
  try
  {
    vector<float> result = ctx->classifier->Classify(buffer, length);
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < result.size(); ++i)
    {
      os << "\"" << result[i] << "\"";
      if (i != result.size() - 1)
        os << ", ";
    }
    os << "]";
    
    errno = 0;
    std::string str = os.str();
    return strdup(str.c_str());
  }
  catch (const std::invalid_argument&)
  {
    errno = EINVAL;
    return NULL;
  }

}

extern "C" void classifier_destroy(classifier_ctx* ctx)
{
    delete ctx->classifier;
    delete ctx;
}

