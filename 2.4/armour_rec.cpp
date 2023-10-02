#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgproc.hpp> 
#include <opencv2/videoio.hpp> 
#include <opencv2/highgui.hpp> 

using namespace cv;

//----------------------------------------预参数-----------------------------------------------------

Mat CAMERA_MADRIX = ( Mat_<double>(3,3) << 1.3859739625395162e+03, 0, 9.3622464596653492e+02, 0,
       1.3815353250336800e+03, 4.9459467170828475e+02, 0, 0, 1);  // 相机内参
Mat DIST_COEFFS = ( Mat_<double>(5,1) << 7.0444095385902794e-02, -1.8010798300183417e-01,
       -7.7001990711544465e-03, -2.2524968464184810e-03,
       1.4838608095798808e-01 );  // 畸变矩阵

double ARMOUR_LENGTH = 100;
double ARMOUR_WIDTH = 60;  // 装甲板长宽

//-------------------------------------预处理到二值化-------------------------------------------

Mat pretreatment(Mat frame){
    
    //-------RBG分离通道（红色）-------
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
    //imshow("baw",dst_baw);

    return dst_baw;
}

//--------------------------------------提取感兴趣区域------------------------------------------------

Mat region_interested(Mat frame, int height, int width){    
    int row, col;
    for(row = 0; row < int(height / 10); ++row)  // 行，即y轴
        {   
            unsigned char* ptr1 = frame.ptr<unsigned char>(row);
            for(col = 0; col < width; ++col)
            {
                unsigned char temp = ptr1[col];
                std::cout<<temp<<std::endl;
                temp = 0;  // 掩码操作，设为黑色
                ptr1[col] = temp;
            }
        }
    //imshow("mask",frame);
    return frame;
}

//-----------------------------------------滤波处理----------------------------------------------------

Mat filtering(Mat frame){
    
    cv::Mat frame_erode;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    cv::erode(frame, frame_erode, kernel, cv::Point(-1,-1), 1);  // 腐蚀一次
    cv::imshow("filtering", frame_erode);

    return frame_erode;
}

//-------------------------------------------主函数--------------------------------------------------

int main(){
    cv::VideoCapture cap("../armour.avi");

    if(cap.isOpened())
    {   
        //-------帧参数-------
        cv::Mat frame;
        int frame_num = 0;  // 记录读取的帧数
        const double CAP_WIDTH = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double CAP_HEIGHT = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        //-------测距参数------
        double center_point[2] = {0,0};  // 用于装甲板中心点的储存
        double rect_centerpoint [4] = {0,0,0,0};  // 用于矩形中心点的储存，x-y-x-y
        double distance_midpoint_to_midpoint[4] = {0,0,0,0};  // 用于记录中心点到矩形中心点的距离,x-y-x-y
        double DISTANCE_DEEP_ZERO = 0;  // 用于记录第一帧（假设深度为零）的中心点image距离
        double deep_cal = 0;  // 用于记录深度
        Point3d world_point_array [5] = {Point3d(0,0,0)};  // 用于聚集世界坐标系点
        Point2d img_point_array [5] = {Point2d(0,0)};  // 用于聚集像素坐标系点
        std::vector<cv::Point3d> world_point;  // 用于PnP输入世界坐标系点集
        std::vector<cv::Point2d> img_point;  // 用于PnP输入像素坐标点集

        //-------循环参数-------
        int i_temp = 0;
        int j_temp = 0;  // 用于循环下标


        //-------开始读取-------
        while(cap.read(frame))  // 读取视频帧
        {   frame_num++;
            if(!frame.empty())
            {   
                //-------预处理至滤波完成-------
                cv::imshow("ori",frame);
                
                Mat baw = pretreatment(frame);
                
                Mat mask = region_interested(baw, CAP_HEIGHT, CAP_WIDTH);

                Mat dst = filtering(mask);

        //----------------------------------轮廓、中心点检测（主要模块1）-------------------------------------------
                
                std::vector<std::vector<cv::Point> > Point_fix;
                std::vector<cv::Vec4i> hierarchy;
                findContours(dst, Point_fix, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);  // 检索外部轮廓，轮廓所有点储存

                //for (size_t i = 0; i < Point_fix.size(); i++) 
		            //cv::drawContours(frame, Point_fix, i, cv::Scalar(255, 255, 255), 1);  //暂时不可用

                //-------画出重要点-------
                std::vector<RotatedRect> minAreaRects(Point_fix.size());  // 用于储存矩形信息，成员center
                double center_x = 0;  
                double center_y = 0;  // 用于求中心点均值
                
                for (i_temp = 0 , j_temp = 0; i_temp < Point_fix.size(); i_temp++)  // 遍历所有轮廓
                {   
		            minAreaRects[i_temp] = minAreaRect(Point_fix[i_temp]);  // 获取轮廓的最小外接矩形
		            if(minAreaRects[i_temp].size.width * minAreaRects[i_temp].size.height >= 30)
                        {   
                        rect_centerpoint[j_temp] = minAreaRects[i_temp].center.x;
                        rect_centerpoint[j_temp+1] += minAreaRects[i_temp].center.y;  // 记录矩形中点的x，y坐标
                        j_temp += 2;
                        circle(frame, minAreaRects[i_temp].center, 10, Scalar(255, 0, 120), -1);
                        //std::cout<<"x = "<< minAreaRects[i].center.x << " y = " <<minAreaRects[i].center.y <<std::endl;
                        //std::cout<<"sum_x = "<<sum_x<<" sum_y = "<<sum_y<<std::endl;
                        }
                }
                
                //-------计算并画出中心点-------
                center_point[0] = double ((rect_centerpoint[0] + rect_centerpoint[2]) / 2);
                center_point[1] = double ((rect_centerpoint[1] + rect_centerpoint[3]) / 2);
                Point2d center_target(center_point[0], center_point[1]);
                circle(frame, center_target, 3, Scalar(255, 0, 255), -1);
                imshow("point",frame);

                //std::cout<<Point_fix.size()<<std::endl;

        //-------------------------------------测距模块--------------------------------------------
                //-------测距预处理-------
                distance_midpoint_to_midpoint[0] = rect_centerpoint[0] - center_point[0];
                distance_midpoint_to_midpoint[1] = rect_centerpoint[1] - center_point[1];
                distance_midpoint_to_midpoint[2] = rect_centerpoint[2] - center_point[0];
                distance_midpoint_to_midpoint[3] = rect_centerpoint[3] - center_point[1];  // 计算中心点间的dx与dy

                for( i_temp = 0 ; i_temp < 4 ; i_temp++ )  // 计算绝对距离
                {
                    if(distance_midpoint_to_midpoint[i_temp] < 0)
                        distance_midpoint_to_midpoint[i_temp] = -distance_midpoint_to_midpoint[i_temp];
                }

                //-------记录第一帧信息，假设其在深度方向上水平-------
                if(frame_num == 1)  // 用于记录第一帧（假设深度为零）的中心点image距离
                    DISTANCE_DEEP_ZERO = ((std::hypot(distance_midpoint_to_midpoint[0],distance_midpoint_to_midpoint[1]))+(std::hypot(distance_midpoint_to_midpoint[2],distance_midpoint_to_midpoint[3])))/2;  
                
                //-------计算世界坐标系下的深度参数-------
                //deep_cal = (std::sqrt(std::pow(DISTANCE_DEEP_ZERO, 2) - (std::pow(distance_midpoint_to_midpoint[0], 2)+std::pow(distance_midpoint_to_midpoint[1],2))) + std::sqrt(std::pow(DISTANCE_DEEP_ZERO, 2) - (std::pow(distance_midpoint_to_midpoint[3], 2)+std::pow(distance_midpoint_to_midpoint[4],2)))) / 2;

                //-------构建世界坐标系与像素坐标系-------
                //world_point_array = { Point3d() , Point3d() , Point3d() , Point3d() , Point3d() , };
                //img_point_array = { Point2d() , Point2d() , Point2d() , Point2d() , Point2d() , };



                //cv::Point2f* img_example_2fpoint = new cv::Point2f(4);  
                //cv::Point3f* world_example_3fpoint = new cv::Point3f(4);
                // 注意free！！！
               
                //minAreaRects[i_memory].points(img_example_2fpoint);  // 2D点坐标
                
                //delete img_example_2fpoint;
                //delete world_example_3fpoint;
                // 如何知道哪个点对应世界坐标系上的对应点？
                //float world_example[12] = {0,0,0,0,20,}
                // 装甲板尺寸是多少？


                //solvePnP()


                
                if(cv::waitKey(1) >= 0)  //一秒约24帧,按下键盘任意键退出
                    break;
            }
        }
    }
    return 0;
}