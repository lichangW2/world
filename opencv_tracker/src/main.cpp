#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

using namespace std;
using namespace cv;

const  string video = "/Users/cj/workspace/world/video_tracker/sources/movie/aircraft.mp4";

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}
Point2f point;
bool addRemovePt = false;
static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
        cout<<"tracking point: %v"<<point<<endl;
    }
}


int main(int argc, char** argv){

    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);
    const int MAX_COUNT = 500;
    bool needToInit = false;
    bool nightMode = false;
    help();

    cap.open(video);
    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
        return 0;
    }
    namedWindow( "LK Demo", 1 );
    setMouseCallback( "LK Demo", onMouse, 0 );
    Mat gray, prevGray, image, frame;
    vector<Point2f> points[2];

    cap.set(CAP_PROP_POS_FRAMES,1680);
    for(;;)
    {
        cap >> frame;
        if( frame.empty() )
            break;
        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);
        if( nightMode )
            image = Scalar::all(0);
        if( needToInit )
        {
            cout<<"need to init"<<endl;
            // automatic initialization
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
            imshow("LK Demo", image);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            addRemovePt = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            cout<<"points 0:"<<points[0].size()<<"len(status):"<<status.size()<<"points 1:"<<points[1].size()<<endl;
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }
                if( !status[i] ){
                    cout<<"points 1 content:"<<points[1][i]<<endl;
                    continue;
                }

                points[1][k++] = points[1][i];
                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
        }
        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }
        needToInit = false;
        imshow("LK Demo", image);
        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch( c )
        {
            case 'r':
                needToInit = true;
                break;
            case 'c':
                points[0].clear();
                points[1].clear();
                break;
            case 'n':
                nightMode = !nightMode;
                break;
        }
        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
    }

    return  0;
}

int kcf(){
    Rect2d roi;
    Rect2d nnroi;
    Mat frame;
    bool md;
    int count=0;
    //根据物体在上一帧的位置预测它在下一帧的位置，但这样会积累误差，而且一旦物体在图像中消失，追踪器就会永久失效，即使物体再出现也无法完成追踪
    Ptr<TrackerKCF> kcftracker= TrackerKCF::create();
    //Ptr<TrackerTLD> tldtracker = TrackerTLD::create();
    VideoCapture cap(video);
    cap.set(CAP_PROP_POS_FRAMES,1680);
    cap >> frame;
    roi=selectROI("tracker",frame);
    if (roi.width==0||roi.height==0){
        cerr<<"invalid roi"<<endl;
        return 0;
    };
    cout << "origin roi: " << roi <<endl;

    kcftracker->init(frame,roi);
    //cap.set(CAP_PROP_POS_FRAMES,0);

    for(;;){
        Mat nframe;
        Rect2d nroi;

        cap>>nframe;
        if (nframe.rows == 0 || nframe.cols == 0){
            cerr<<"invalid frame, the video may over"<<endl;
            break;
        }
        md=kcftracker->update(nframe, nroi);
        if (md){
            cout<<"here............." <<endl;
            nnroi.x=nroi.x;
            nnroi.y=nroi.y;
            nnroi.width=nroi.width;
            nnroi.height=nroi.height;
            count=25;
        }

        cout<< "new roi: "<< nroi << "md:"<<md <<endl;
        if (count>=0){
            rectangle(nframe, nnroi, Scalar(255, 0, 0), 2, 1);
            count--;
        }

        imshow("tracker", nframe);

        if (waitKey(1) == 27)
            break;
    }
    return 0;
}