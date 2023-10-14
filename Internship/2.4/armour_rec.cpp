#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp> 
#include <opencv2/videoio.hpp> 
#include <opencv2/highgui.hpp> 

#include "armour_tool.h"

using namespace cv;


//----------------------------------------预参数----------------------------------------------------
const Mat CAMERA_MADRIX = ( Mat_<double>(3,3) << 1400,0,360,0,1400,240,0,0,1);  // 相机内参
const Mat DIST_COEFFS = ( Mat_<double>(1,5) << 0, 0, 0, 0, 0 );  // 畸变矩阵

const double ARMOUR_LENGTH = 200;
const double ARMOUR_WIDTH = 100;  // 装甲板长宽

//-------测距模式选择----
    const char RECT_MODE = 1;  // 矩形检测
    const char RIOH_MODE = 2;  // 菱形检测
//----------------------------------------主函数-----------------------------------------------------

int main()
{
    cv::VideoCapture cap("../armour.avi");

    if(cap.isOpened())
    {   
        //-------帧参数-------
        cv::Mat frame;
        int frame_num = 0;  // 记录读取的帧数
        const double CAP_WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double CAP_HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        //-------测距参数------
        float last_rect_angle[10] = {0};  // 储存上一帧角度信息，光流法可能用到

        //-------开始读取-------
        while(cap.read(frame))
        {   
            frame_num++;
            if(!frame.empty())
            {   
                //-------预处理至滤波完成-------
                //cv::imshow("ori",frame);
                
                Mat baw = pretreatment(frame);
                Mat mask = region_interested(baw, CAP_HEIGHT, CAP_WIDTH);
                Mat dst = filtering(mask);
                
                //-------检索外部轮廓-------
                std::vector<std::vector<cv::Point> > Point_fix;
                std::vector<cv::Vec4i> hierarchy;
                findContours(dst, Point_fix, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                //drawContours(frame, Point_fix, -1, Scalar(255, 0, 255), -1);
                //imshow("drawframe",frame);

                //-------轮廓、中心点检测（主要模块1)-------
                std::vector<std::vector<Point2f>> Rect_angpoint;
                Rect_angpoint = direct_rect(Point_fix, frame, RECT_MODE);
                
                for(int i = 0; i < Rect_angpoint.size(); i++)
                    draw_rect(Rect_angpoint[i], frame);
                imshow("Rect",frame);
                //draw_point(frame, center_point, rect_centerpoint);
                
                //-------测距、模式选择----
                for(int i = 0; i < Rect_angpoint.size(); i++)
                    distances_detect(Rect_angpoint[i], frame, RECT_MODE);

            }
        if(cv::waitKey(50) >= 0)  //一秒约24帧,按下键盘任意键退出
            break;
        }
    }
    return 0;
}


