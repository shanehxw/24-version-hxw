#include <iostream>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui.hpp>
using namespace std;

int main(){
    //cv::VideoCapture cap(0);
    cv::VideoCapture cap("../test.avi");
    if(cap.isOpened())
        
        cout<<"1"<<endl;
        while(1)
        {
            cv::Mat frame;
            cap.read(frame);
            if(!frame.empty())
                cv::imshow("video",frame);
                cv::waitKey(42);  // 一秒24帧
        }

    return 0;
}