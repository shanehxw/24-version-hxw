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
        cv::Mat frame;
        int frame_num = 0;  // 记录读取的帧数
        //-------视频参数-------
        const double CAP_WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double CAP_HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        //-------测距参数------
        float last_rect_angle[10] = {0};  // 储存上一帧角度信息，预测可能用到

        //-------开始读取-------
        while(cap.read(frame))
        {   
            frame_num++;
            if(!frame.empty())
            {   
                //-------预处理至滤波完成-------
                //cv::imshow("ori",frame);
                
                // BGR分离R通道二值化
                Mat baw = pretreatment(frame);  
                // 掩码掩盖上方灯条
                Mat mask = region_interested(baw, CAP_HEIGHT, CAP_WIDTH);
                // 2次膨胀增大装甲板亮光区域，防止装甲板因过于纤细造成的矩形识别误差
                Mat dst = filtering(mask);  
                
                //-------检索外部轮廓-------
                std::vector<std::vector<cv::Point> > Point_fix;
                std::vector<cv::Vec4i> hierarchy;
                findContours(dst, Point_fix, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                //drawContours(frame, Point_fix, -1, Scalar(255, 0, 255), -1);
                //imshow("drawframe",frame);

                //-------轮廓、中心点检测（主要模块1)-------
                //模块说明：
                //direct_rect()--轮廓预处理--对轮廓使用minarearect（），通过长宽比初筛矩形，对筛选出来的矩形从左到右进行排序
                //combine_rect():--对一组矩形的参数进行计算，方便接下来操作--求出装甲板大致长宽、倾斜角，并进行模式选择
                //1、采用矩形检测方式：
                //  rect_point_cal()--通过几何关系，利用装甲板中心点计算装甲板四个角点
                //  rect_amend_for_width()--通过较为稳定的灯条两中心点斜率，保留角点间的中心点，通过修正装甲板斜率，重新计算装甲板角点坐标
                //2、采用菱形检测方式：
                //  rhom_point_cal()--通过几何关系，计算装甲板长宽线段中心的四个特征点
                std::vector<std::vector<Point2f>> Rect_angpoint;
                Rect_angpoint = direct_rect(Point_fix, frame, RECT_MODE);
                
                // 绘制特征点连线
                for(int i = 0; i < Rect_angpoint.size(); i++)
                    draw_rect(Rect_angpoint[i], frame);
                imshow("Rect",frame);
                //draw_point(frame, center_point, rect_centerpoint);
                
                //-------测距、模式选择----
                //以装甲板中心为世界坐标系原点，因此平移矩阵第三个参数即是图像深度
                for(int i = 0; i < Rect_angpoint.size(); i++)
                    distances_detect(Rect_angpoint[i], frame, RECT_MODE);

            }
        if(cv::waitKey(50) >= 0)  //一秒约24帧,按下键盘任意键退出
            break;
        }
    }
    return 0;
}


