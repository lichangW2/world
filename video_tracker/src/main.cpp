#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"

using namespace std;
using namespace cv;

const string DefaultMovie("/Users/cj/workspace/project/video_tracker/sources/movie/aircraft.mp4");
const string SamplePage("/Users/cj/workspace/project/video_tracker/sources/sample.png");

class A{
public:
    A(){
        cout<< "in A"<<endl;
    };
};

class B{
public:
    B(A &a){
      cout <<"in b"<<endl;
    };
};

void test(B b){
    return;
}


int main(int argc,char* argv[]) {

    string movie;

    if (argc>1){
        movie = argv[1];
    }else{
        movie = DefaultMovie;
        clog << "no movie provided,use: "<< DefaultMovie <<endl;
    }

    bool HOG = true;
    bool FIXEDWINDOW = true;
    bool MULTISCALE = true;
    bool SILENT = true;
    bool LAB = true;

    for(int i = 2; i <  argc; i++){
        if ( strcmp (argv[i], "hog") == 0 )
            HOG = true;
        if ( strcmp (argv[i], "fixed_window") == 0 )
            FIXEDWINDOW = true;
        if ( strcmp (argv[i], "singlescale") == 0 )
            MULTISCALE = false;
        if ( strcmp (argv[i], "show") == 0 )
            SILENT = false;
        if ( strcmp (argv[i], "lab") == 0 ){
            LAB = true;
            HOG = true;
        }
        if ( strcmp (argv[i], "gray") == 0 )
            HOG = false;
    }

    VideoCapture cap(movie);
    if(!cap.isOpened())
        return -1;

    cout << "video frame rate: "<< cap.get(cv::CAP_PROP_FPS ) <<endl;
    cout << "video total frames: " << cap.get( cv::CAP_PROP_FRAME_COUNT) <<endl;
    cout << "video frame mat dims: " << cap.get(cv::CAP_PROP_FORMAT) <<endl;

    Mat edges;
    namedWindow("edges",1);

    Mat sample_frame;
    cap.set(CAP_PROP_POS_FRAMES,1680);
    cap.read(sample_frame);
    KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    tracker.init(Rect(300,45,50,80),sample_frame);

    //rectangle(sample_frame,Rect(300,45,50,80),Scalar(255,0,0));
    //imshow("edges",sample_frame);
    //if (waitKey(0)>0){

    //}

    cap.set(CAP_PROP_POS_FRAMES,1680);
    for(;;){

        Mat frame;
        Rect rect;

        cap>>frame;
        //cvtColor(frame,edges);
        //GaussianBlur(edges,edges,Size(7,7),1.5,1.5);
        //Canny(edges,edges,0,30,3);

        //cout << frame.size() << endl;
        //cout <<"dims:"<< frame.dims << endl;
        //cout <<"channels:"<<frame.channels() << endl;
        //cout << frame <<endl;

        cout << "current position: " << cap.get(CAP_PROP_POS_FRAMES) <<endl;
        //imwrite(SamplePage,frame);
        //rectangle(frame,Rect(85,18,135,280),Scalar(255,0,0));


        rect=tracker.update(frame);
        rectangle(frame,rect,Scalar(255,0,0));
        imshow("edges",frame);
        if (waitKey(30)>0){
            break;
        }
    }
    cap.release();
    //return 0;
}