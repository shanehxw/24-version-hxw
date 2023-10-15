#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp> 
//#include <opencv2/videoio.hpp> 
//#include <opencv2/highgui.hpp> 

#include "pm_rec_tool.h"

using namespace cv;

//------------------------------------------------参数----------------------------------------------------

//-------相机参数-------
const double VIDEO_WIDTH = 1280;
const double VIDEO_HEIGHT = 1024;  // 视频尺寸

const Mat CAMERA_MADRIX = ( Mat_<double>(3,3) << 1400,0,360,0,1400,240,0,0,1);  // 相机内参
const Mat DIST_COEFFS = ( Mat_<double>(1,5) << 0, 0, 0, 0, 0 );  // 畸变矩阵

//-------世界参数-------
const double R_PM_LARGE = 1400;
const double R_PM_SMALL = 300;
const double ANGLE_PMDISTANCE = 72;

const double DISTANCE_ROBOT = 700;
const double HEIGHT_PM = 1550;
const double ANGLE_ROBOT_TO_PM = 2;

const double PI = 3.14159265;

//-------------------------------------------------3.1---------------------------------------------------

//-------通道分离主函数-------
Mat img_baw(Mat frame)  // 对两个二值图取交集
{
     //分别作灰度图和通道分离处理
    Mat img_gray_baw = img_gray(frame);
    Mat img_channel_baw = img_channel(frame);

    //对两个二值化图像进行交运算,降低噪声
    Mat img_and;
    bitwise_and(img_gray_baw, img_channel_baw, img_and);
    imshow("img_and",img_and);
    
    return img_and;
}

//-------判断是否为激活靶心-------
std::vector<Point2f> judge_target(std::vector<Point2f> rect_center, std::vector<Point2f> pre_target_point, Point2f center_point_R, Mat frame)
{   
    std::vector<Point2f> target;
    // 计算预估配对点
    std::vector<Point2f> judge_points;
    float k = 0.477;
    for(int i = 0; i < pre_target_point.size(); i++){
        float judge_x = center_point_R.x + (pre_target_point[i].x - center_point_R.x) * k;
        float judge_y = center_point_R.y + (pre_target_point[i].y - center_point_R.y) * k;
        Point2f temp_judge(judge_x, judge_y);
        judge_points.push_back(temp_judge);
    }
    //draw_point(judge_points, frame, Scalar(255,0,0), 3);
    //imshow("judge", frame);

    int i = 0;
    int j = 0;
    std::vector<Point2f> mid_point;
    //计算各矩形中心点的中心点
    for(i = 0; i < rect_center.size(); i++){
        for(j = 0; j < rect_center.size(); j++){
            if(i != j){
                float rect_dis = calculateDistance(rect_center[i], rect_center[j]);
                if(rect_dis < 80){
                    Point2f temp_point ( (rect_center[i].x + rect_center[j].x)/2 , (rect_center[i].y + rect_center[j].y)/2 );
                    mid_point.push_back(temp_point);
                }
            }
        }
    }
    //draw_point(mid_point, frame, Scalar(0,255,0), 3);

    //利用距离阈值判断目标点是否与激活矩阵匹配,对距离在near_dis以内的进行计算
    float near_dis = 15;
    for(i = 0; i < judge_points.size(); i++){
        for(j = 0; j < mid_point.size(); j++){
            if(i != j){
                float dis = calculateDistance(mid_point[j], judge_points[i]);
                if(dis < near_dis){
                    if(VIDEO_WIDTH/10 < pre_target_point[i].x && VIDEO_HEIGHT/10 < pre_target_point[i].y){
                        target.push_back(pre_target_point[i]);
                        //circle(frame, judge_points[i], 3, Scalar(255,120,255), -1);
                        //circle(frame, mid_point[j], 3, Scalar(0,0,255), -1);
                        //break;
                    }
                }
            }
        }
    }
    return target;
}

//-------靶心检测主函数-------
std::vector<Point2f> Point_detect_circle(Mat dst, Mat frame, Point2f center_R)
{   
    //不用轮廓变化的滤波
    dst = img_filter_forcircle1(dst);

    //用轮廓变化的滤波
    //cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    //cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    //cv::GaussianBlur(dst, dst, Size(3,3), 0, 0);
    //cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    //cv::dilate(dst, dst, kernel3, Point(-1,-1), 1);
    //cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    //cv::GaussianBlur(dst, dst, Size(3,3), 0, 0);
    //cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    //cv::GaussianBlur(dst, dst, Size(3,3), 0, 0);
    //imshow("erode", dst);

    //-------轮廓检测-------
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    //使用轮廓变化
    //findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    //contours = find_target_contours0(contours, hierarchy);

    //-------轮廓检测调试
    //cv::Mat empty_Mat = cv::Mat::zeros(frame.size(), frame.type());
    //drawContours(empty_Mat, contours, -1, Scalar(255, 0, 255), -1);
    //imshow("drawframe",empty_Mat);

    std::vector<Point2f> cir_center_pretreat;
    std::vector<RotatedRect> rect_to_cir;

    for (int i = 0; i < contours.size(); i++) 
    {
        // 计算轮廓的矩
        Moments M = moments(contours[i]);
        // 计算轮廓的质心坐标
        Point2f center(M.m10 / M.m00, M.m01 / M.m00);
        
        // 剔除R标周围点
        float R_roi = 60;
        if(center_R.x - R_roi < center.x && center.x < center_R.x + R_roi && center_R.y - R_roi < center.y && center.y < center_R.y + R_roi)  // 剔除R标附近的点，80*80的矩形
            continue;

        // 计算最小圆形
        RotatedRect temprect_to_cir = minAreaRect(contours[i]);
        float temprect_width_height[2];
        find_width_and_height(temprect_to_cir.size.width, temprect_to_cir.size.height, temprect_width_height);
    
        // 判断最小圆形半径是否合理（可以根据实际需求进行调整）
        if (temprect_width_height[0] / temprect_width_height[1] < 1.8) 
        {    
            float radius = std::sqrt(temprect_width_height[0] * temprect_width_height[0] + temprect_width_height[1] * temprect_width_height[1]) / 2;
            if( radius > 3 && radius < 40)
            {   
                cir_center_pretreat.push_back(temprect_to_cir.center);
                cir_center_pretreat.push_back(center);

            }
        }
    }
    //draw_point(cir_center_pretreat, frame, Scalar(0,255,255), 3);
    //imshow("cir_center", frame);

    // 点的分类与聚合
    std::vector<Point2f> cir_center_point = point_fix(cir_center_pretreat, 160, 190);

    draw_point(cir_center_point, frame, Scalar(255,255,255), 5);
    imshow("cir_center", frame);
    
    //std::cout<<"---------------------------"<<std::endl;
    
    return cir_center_point;

}

//-------矩形检测主函数-------
std::vector<Point2f> Point_detect_rect(Mat dst, Mat frame, Point2f* center_point)  // 要求膨胀至闭环才能使用， 主函数
{   
    //-------简单滤波-------
    dst = img_filter_forrect(dst);

     //-------轮廓检测-------
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    //-------轮廓检测调试
    //cv::Mat empty_Mat = cv::Mat::zeros(frame.size(), frame.type());
    //drawContours(empty_Mat, contours, -1, Scalar(255, 0, 255), -1);
    //imshow("drawframe",empty_Mat);

    std::vector<Point2f> rect_center = rect_contours_pre_recognize(contours, frame, center_point);
    // circle(frame, center_point, 1, Scalar(255,255,255), -1 );
    

    //imshow("judge", frame);
    //rect_compare(contours_min_rects, frame, judge_points);

    return rect_center;
}

//-------角度检测主函数-------
std::vector<float> world_angle_cal(Point2f center_R, std::vector<Point2f> target_points)
{
    std::vector<float> angles;
    for(int i = 0; i < target_points.size(); i++){
        if(target_points[i].x != center_R.x){  // 若存在斜率y/x
            float temp_angle = -((std::atan2(target_points[i].y - center_R.y, target_points[i].x - center_R.x)) * 180 / PI);  // 计算世界坐标系下的角度
            angles.push_back(temp_angle);
        }
        else{  // 若不存在斜率y/x
            if(target_points[i].y > center_R.y){
                float temp_angle = -90;
                angles.push_back(temp_angle);
            }
            else if(target_points[i].y <= center_R.y){
                float temp_angle = 90;
                angles.push_back(temp_angle);
            }
        }
    }
    return angles;
}

std::vector<float> angles_detect(Point2f center_R, std::vector<Point2f> target_points, Mat frame)
{
    std::vector<float> angles = world_angle_cal(center_R, target_points);  // 计算出角度，序号和target序号一一对应
    std::vector<std::string> put_txt;
    for(int i = 0; i < angles.size(); i++){
        std::string s = std::to_string(angles[i]);
        put_txt.push_back(s);
    }
    int j = 0;
    for(int txt_y = 20, j = 0; j < angles.size(); j++, txt_y += 20){
        putText(frame, put_txt[j], cv::Point(20, txt_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    //std::cout<<angles.size()<<std::endl;

    return angles;
}
//-------------------------------------------------3.3----------------------------------------------------

//------------------------------------------------主函数--------------------------------------------------
int main(){
    std::cout<<"Hello,pm"<<std::endl;

    cv::VideoCapture cap("../Power_machine.avi");

    Mat frame;
    int flame_num = 0;

    if(cap.isOpened())
    {
        while(cap.read(frame))
        {   
            flame_num++;
            //imshow("ori",frame);

            //-------3.1-------
            Mat dst_baw = img_baw(frame);

            Point2f center;  // R标位置
            std::vector<Point2f> rect_center = Point_detect_rect(dst_baw, frame, &center);  // 激活矩形中心点
            circle(frame, center, 5, Scalar(255,255,255), -1);
            imshow("pre_target", frame);

            std::vector<Point2f> pre_target = Point_detect_circle(dst_baw, frame, center);  // 靶心中心点

            std::vector<Point2f> target;
            if(&center != NULL)
                target = judge_target(rect_center, pre_target, center, frame);
            draw_point(target, frame, Scalar(255,120,255), 5);
            std::cout<<target.size()<<std::endl;
            //imshow("target", frame);

            std::vector<float> angles = angles_detect(center, target, frame);
            imshow("3.1", frame);
            //-------3.1-------
            
            //-------3.3-------


            if(cv::waitKey(50) >= 0)
                break;
        }


    }

    return 0;
}