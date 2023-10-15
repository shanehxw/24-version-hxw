#pragma once

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp> 
//#include <opencv2/videoio.hpp> 
//#include <opencv2/highgui.hpp> 

using namespace cv;

// 对minareaRect的宽高进行排序
void find_width_and_height(float width, float height, float* array);

// 通过Point2f[4]画出矩形
void draw_rect_array(Point2f rect_point[], Mat frame, Scalar color);

// 通过vector<Point2f>画出矩形
void draw_rect(std::vector<Point2f> rect, Mat frame);  

// 通过std::vector<Point2f>画出点
void draw_point(std::vector<Point2f> points, Mat frame, Scalar color, int r);  

// 通过两个Point2f计算距离
float calculateDistance(cv::Point2f pt1, cv::Point2f pt2);   

// 通过中心点对矩形进行从左到右的排序
std::vector<RotatedRect> compare_Left_or_Right(const RotatedRect rect_1, const RotatedRect rect_2);  

// 灰度图转化
Mat img_gray(Mat frame);  

// BGR通道处理
Mat img_channel(Mat frame);  

// 滤波处理（矩形膨胀）
Mat img_filter_forrect(Mat dst);  

// 滤波处理（靶心腐蚀）
Mat img_filter_forcircle1(Mat dst);  

//在轮廓中搜索，返回只有一级子轮廓以及没有子轮廓的轮廓
std::vector<std::vector<Point>> find_target_contours0_1(std::vector<std::vector<Point>> contours, std::vector<cv::Vec4i> hierarchy);

// 寻找没有子轮廓的轮廓
std::vector<std::vector<Point>> find_target_contours0(std::vector<std::vector<Point>> contours, std::vector<cv::Vec4i> hierarchy);

// 邻近点取方差聚合(初步)
Point2f near_fix(std::vector<Point2f> group, int r);

//-------简单聚合，消除非常近的点-------
std::vector<Point2f> point_fix_simple(std::vector<Point2f> points);

// 聚合主函数
std::vector<Point2f> point_fix(std::vector<Point2f> points, float k1, float k2);

// 矩形和R标检测
std::vector<Point2f> rect_contours_pre_recognize(std::vector<std::vector<cv::Point> > contours, Mat frame, Point2f* center_point);


// 进行两两分组  // 目前来看，由于检测矩形的不稳定，两两分组不太能对靶心检测进行辅助，因此暂时不使用
std::vector<std::vector<RotatedRect>> rect_compare(std::vector<cv::RotatedRect> pre_minrect, Mat frame, std::vector<Point2f> judge_point);






