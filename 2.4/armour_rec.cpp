#include <iostream>
#include <vector>
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp> 
#include <opencv2/videoio.hpp> 
#include <opencv2/highgui.hpp> 



int main(){
    cv::VideoCapture cap("../armour.avi");
    
    if(cap.isOpened())
    {
        cv::Mat frame;
        while(cap.read(frame))  // 读取视频帧
        {
            if(!frame.empty())
            {   
                //cv::imshow("ori",frame);
                //-------分离通道（红色）-------
                //cv::cvtColor(frame, RBG_frame, int code, int dstCn=0 );
                std::vector<cv::Mat> channels;
                //cv::Mat imageBlueChannel;
                //cv::Mat imageGreenChannel;
                cv::Mat imageRedChannel;  // 测试得red channel单独时最佳
                
                cv::split(frame, channels);
                //imageBlueChannel = channels.at(0);
                //imageGreenChannel = channels.at(1);
                imageRedChannel = channels.at(2);

                //cv::imshow("【BlueChannel】", imageBlueChannel);
                //cv::imshow("【GreenChannel】", imageGreenChannel);
                //cv::imshow("【RedChannel】", imageRedChannel);

                //-------二值化-------
                cv::Mat dst_baw;
                cv::threshold(imageRedChannel, dst_baw, 220, 255, cv::THRESH_BINARY);

                //-------滤波处理-------
                cv::Mat dst;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
                cv::erode(dst_baw, dst, kernel, cv::Point(-1,-1), 1);
                cv::imshow("video_dst_baw",dst_baw);

                //-------检测轮廓-------
                std::vector<std::vector<cv::Point> > Point_fix;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(dst, Point_fix, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);  // 检索外部轮廓，轮廓所有点储存

                //for (size_t i = 0; i < Point_fix.size(); i++) 
		            //cv::drawContours(dst, Point_fix, -1, cv::Scalar(0, 255, 0), i);

                cv::imshow("video",dst);
                if(cv::waitKey(42) >= 0)  //一秒约24帧,按下键盘任意键退出
                    break;
            }
        }
    }
    return 0;
}