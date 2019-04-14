#include <iostream>
#include <ctime>
#include <vector>
#include "classification.h"

using std::string;
int main(int argc, char** argv) {
  clock_t start_time1,end_time1,start_time2,end_time2;

  const char* model_file = "/workspace/serving/models/online-model/used/fine_deploy.prototxt";
  const char* trained_file = "/workspace/serving/models/online-model/used/fine_weight.caffemodel";
  start_time1 = clock();
  //Classifier classifier;
  classifier_ctx *ctx = classifier_initialize(
             model_file, trained_file, 1);
  end_time1 = clock();
  double seconds1 = (double)(end_time1-start_time1)/CLOCKS_PER_SEC;
  std::cout<<"init time="<<seconds1<<"s"<<std::endl;
  if (ctx == NULL){
    std::cout << "init net error" << std::endl;
    return -1;
  }

  
  std::vector<string> files = {
    "/workspace/serving/models/image/bomb.jpg",
    "/workspace/serving/models/image/sexy2.jpg",
    "/workspace/serving/models/image/xi_chuan.jpg",
    "/workspace/serving/models/image/illegal_flag.jpg"
  };
  
start_time1 = clock();
for(int j=0; j < 100; ++j){
  for (int i =0; i < files.size(); ++i){
    string file = files[i];
    start_time2 = clock();
    const char* result = classifier_classify(ctx, file.c_str(), file.length());
    end_time2 = clock();
    double seconds2 = (double)(end_time2-start_time2)/CLOCKS_PER_SEC;
    //std::cout<<"--------------->classify time="<<seconds2<<"s"<<std::endl;
    //std::cout << result << std::endl;
  }
}
end_time1 = clock();
seconds1 = (double)(end_time1-start_time1)/CLOCKS_PER_SEC;
std::cout<< 100 * files.size() << " loops: "<<seconds1<<"s"<<std::endl;
/*
  string file = "/workspace/serving/models/image/bomb.jpg";
  //cv::Mat img = cv::imread(file, -1);
  start_time2 = clock();
  //vector<float> result = classifier.Classify(img);
  const char* result = classifier_classify(ctx, file.c_str(), file.length());
  end_time2 = clock();
  double seconds2 = (double)(end_time2-start_time2)/CLOCKS_PER_SEC;
  std::cout<<"--------------->classify time="<<seconds2<<"s"<<std::endl;
  std::cout << result << std::endl;
*/
/*
  string file2 = "/workspace/serving/models/image/sexy2.jpg";
  cv::Mat img2 = cv::imread(file2, -1);
  start_time2 = clock();
  vector<float> result2 = classifier.Classify(img2);
  end_time2 = clock();
  seconds2 = (double)(end_time2-start_time2)/CLOCKS_PER_SEC;
  std::cout<<"--------------->classify time="<<seconds2<<"s"<<std::endl;


  string file3 = "/workspace/serving/models/image/xi_chuan.jpg";
  cv::Mat img3 = cv::imread(file3, -1);
  start_time2 = clock();
  vector<float> result3 = classifier.Classify(img3);
  end_time2 = clock();
  seconds2 = (double)(end_time2-start_time2)/CLOCKS_PER_SEC;
  std::cout<<"--------------->classify time="<<seconds2<<"s"<<std::endl;

  for (int i = 0; i < result3.size(); ++i) {
    if (result3[i] >0.000000000000000001)
      std::cout << i << "  "<< result3[i] << std::endl;
  }

  string file4 = "/workspace/serving/models/image/illegal_flag.jpg";
  cv::Mat img4 = cv::imread(file4, -1);
  clock_t start_time4 = clock();
  vector<float> result4 = classifier.Classify(img4);
  clock_t end_time4 = clock();
  double seconds4 = (double)(end_time4-start_time4)/CLOCKS_PER_SEC;
  std::cout<<"--------------->classify time="<<seconds4<<"s"<<std::endl;

  for (int i = 0; i < result4.size(); ++i) {
    if (result4[i] >0.000000000000000001)
      std::cout << i << "  "<< result4[i] << std::endl;
  }
*/
}
