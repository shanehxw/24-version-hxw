#pragma once

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

using namespace cv;

// 求Point中心点
cv::Point2f mid_point(Point2f a, Point2f b);

// 把float[2]转化成Point
Point2f array_to_point(float* array);

// 对minareaRect的宽高进行排序
void find_length_and_width(float width, float height, float* array);

// 通过Point2f[4]画出矩形
void draw_rect_array(Point2f rect_point[], Mat frame);  

// 通过vector<Point2f>画出矩形
void draw_rect(std::vector<Point2f> rect, Mat frame);  

// 通过中心点对矩形进行从左到右的
bool compare_Left_or_Right(const RotatedRect& pt1, const RotatedRect& pt2);

// compare_Left_or_Right的触发函数
std::vector<RotatedRect> pointrank1(std::vector<RotatedRect> Rect_point);

// 对点从下到上排序
bool compareValue(const Point2f& pt1, const Point2f& pt2);

// compareValue的触发函数
std::vector<Point2f> pointrank(std::vector<Point2f> Rect_point);  

// 预处理到二值化
Mat pretreatment(Mat frame);  

// 掩码处理上方灯条 --可以完善，通过对上一帧的装甲板识别确定装甲板大致区域，降低干扰
Mat region_interested(Mat frame, int height, int width);  

// 通过2次膨胀增大装甲板亮光区域，防止装甲板因过于纤细造成的矩形识别误差
Mat filtering(Mat frame);  

// 对得到的一组矩形修正：取中间点，按照斜率重新计算角点
std::vector<Point2f> rect_amend_for_width(float* array1, float* array2, float k, float avg_width);  

// 使用长度计算矩形角点，包括初步处理以及rect_amend_for_width修正
std::vector<Point2f> rect_point_cal(float avg_dia, float avg_width, float k1_world, float k2_armour, Point2f center);  

// 想法：计算菱形角点，在计算速度上会比计算矩形更加快 --已经实现，但在框选上有误差，可以再调试
std::vector<Point2f> rhom_point_cal(float avg_length, float avg_width, float x_diff, float y_diff, Point2f leftpoint, Point2f rightpoint,Point2f center);

// 通过对长宽以及角度的相似程度，对多组得到的rectage进行装甲板识别 --可以通过设置识别顺序等，减少运行时间，进一步完善
int rect_compare(float rect1_angle, float* rect1_landw, float rect2_angle, float* rect2_landw);

// 对一组矩形进行合并，实现对装甲板的框选，本函数内部是一些参数的初步处理，通过引用其他函数实现功能
std::vector<Point2f> combine_rect(RotatedRect rect1, RotatedRect rect2, Mat frame, char mode);

//轮廓检测主函数
std::vector<std::vector<Point2f>> direct_rect(std::vector<std::vector<cv::Point> > Point_fix, Mat frame, char mode = 1);

//距离识别主函数
void distances_detect(std::vector<Point2f> rect_combine, Mat frame, char mode = 1);
