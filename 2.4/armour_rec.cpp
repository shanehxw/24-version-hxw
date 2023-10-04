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


//----------------------------------------预参数-----------------------------------------------------

const Mat CAMERA_MADRIX = ( Mat_<double>(3,3) << 1400,0,360,0,1400,240,0,0,1);  // 相机内参
const Mat DIST_COEFFS = ( Mat_<double>(1,5) << 0, 0, 0, 0, 0 );  // 畸变矩阵

const double ARMOUR_LENGTH = 200;
const double ARMOUR_WIDTH = 100;  // 装甲板长宽

//----------------------------------------简单工具-----------------------------------------------------------------

cv::Point2f mid_point(Point2f a, Point2f b)
{
    Point2f mid (a.x + b.x , a.y + b.y);
    return mid;
}


Point2f array_to_point(float* array){
    
    Point2f point(array[0],array[1]);
    return point;

}

//--------------------------------------------矩形工具----------------------------------------------------------------------

void find_length_and_width(float width, float height, float* array)
{   
    if(width < height) 
    {
        array[0] = height;
        array[1] = width;
    }
    else
    {
        array[0] = width;
        array[1] = height;
    }
}

void draw_rect_array(Point2f rect_point[], Mat frame){

    for(int i = 0; i < 4; i++)
    {  
        line(frame, rect_point[i%4], rect_point[(i+1)%4], Scalar(255,0,120), 1);
    }
}

void draw_rect(std::vector<Point2f> rect, Mat frame)
{   
    for(int i = 0; i < rect.size(); i++)
    {   
        line(frame, rect[i%rect.size()], rect[(i+1)%rect.size()], Scalar(255,0,120), 1);
    }
}


//-------------------------------------------排序工具---------------------------------------------
bool compare_Left_or_Right(const RotatedRect& pt1, const RotatedRect& pt2)
{
	if (pt1.center.x != pt2.center.x)
		return pt1.center.x < pt2.center.x;  // x从小到大排序
	else
		return pt1.center.y < pt2.center.y;
}

std::vector<RotatedRect> pointrank1(std::vector<RotatedRect> Rect_point)
{  

        std::sort(Rect_point.begin(), Rect_point.end(), compare_Left_or_Right);

        return Rect_point;
}

bool compareValue(const Point2f& pt1, const Point2f& pt2)
{
	if (pt1.y != pt2.y)
		return pt1.y > pt2.y;  // y从小到大排序
	else
		return pt1.x > pt2.x;
}

std::vector<Point2f> pointrank(std::vector<Point2f> Rect_point){  

        std::sort(Rect_point.begin(), Rect_point.end(), compareValue);

        return Rect_point;
}


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
    
    Mat frame_erode;
    Mat frame_dilate;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    cv::erode(frame, frame_erode, kernel, Point(-1,-1), 1);  // 腐蚀一次
    cv::dilate(frame_erode, frame_dilate, kernel, Point(-1,-1), 2);  //膨胀两次
    cv::imshow("filtering", frame_dilate);

    return frame_dilate;
}


//------------------------------------------轮廓检索1.6------------------------------------------------------------
//问题：compare太卡了

std::vector<Point2f> rect_amend_for_width(float* array1, float* array2, float k, float avg_width)  // 取中间点，按照斜率重新计算
{
    float mid_x = (array1[0] + array2[0]) / 2;
    float mid_y = (array1[1] + array2[1]) / 2;

    float mid_distance = avg_width/2;
    float x_distance = mid_distance / std::sqrt(std::pow(k,2)+1);
    float y_distance = k * mid_distance / std::sqrt(std::pow(k,2)+1);  // 再次使用平方和的平方根，或许产生计算误差

    std::vector<Point2f> left_right_point;
    Point2f left_point (mid_x - x_distance, mid_y - y_distance);
    Point2f right_point (mid_x + x_distance, mid_y + y_distance);
    
    left_right_point.push_back(left_point);
    left_right_point.push_back(right_point);

    return left_right_point;

}

std::vector<Point2f> rect_point_cal(float avg_dia, float avg_width, float k1_world, float k2_armour, Point2f center)
{
    float k3 = 0;
    float k4 = 0;
    float ang_to_center_x_1 = 0;
    float ang_to_center_y_1 = avg_dia;
    float ang_to_center_x_2 = 0;
    float ang_to_center_y_2 = avg_dia;
    if(k1_world*k2_armour != 1)
    {
        k3 = (k1_world+k2_armour)/(1-k1_world*k2_armour);  //对k1k2合角,+ 
        ang_to_center_x_1 = avg_dia / std::sqrt(std::pow(k3, 2) + 1);
        ang_to_center_y_1 = k3 * avg_dia / std::sqrt(std::pow(k3, 2) + 1);
    }
    if(k1_world*k2_armour != -1)
    {
        k4 = (k2_armour-k1_world)/(1+k2_armour*k1_world);  // 对k1k2合角，-
        ang_to_center_x_2 = avg_dia / std::sqrt(std::pow(k4, 2) + 1);
        ang_to_center_y_2 = k4 * avg_dia / std::sqrt(std::pow(k4, 2) + 1);
    }

    float RightDown[2] = {center.x + ang_to_center_x_1, center.y + ang_to_center_y_1};  // 右下点初始值
    float LeftDown[2] = {center.x - ang_to_center_x_2, center.y + ang_to_center_y_2};  // 左下点初始值
    float LeftUp[2] = {center.x - ang_to_center_x_1, center.y - ang_to_center_y_1};  // 左上点初始值
    float RightUp[2] = {center.x + ang_to_center_x_2, center.y - ang_to_center_y_2};  // 右上点初始值
    
    std::vector<Point2f> uppoint_left_right;
    std::vector<Point2f> downpoint_left_right;

    downpoint_left_right = rect_amend_for_width(LeftDown, RightDown, k1_world, avg_width);
    uppoint_left_right = rect_amend_for_width(LeftUp, RightUp, k1_world, avg_width);

    Point2f RightDown_point (downpoint_left_right[1]);
    Point2f LeftDown_point (downpoint_left_right[0]);
    Point2f LeftUp_point (uppoint_left_right[0]);
    Point2f RightUp_point (uppoint_left_right[1]);


    std::vector<cv::Point2f> rect_combine;


    rect_combine.push_back(RightDown_point);
    rect_combine.push_back(LeftDown_point);
    rect_combine.push_back(LeftUp_point);
    rect_combine.push_back(RightUp_point);  // ***右下-左下-左上-右上***

    return rect_combine;
}






int rect_compare(float rect1_angle, float* rect1_landw, float rect2_angle, float* rect2_landw)
{   

    if( (rect1_landw[0]-rect2_landw[0]) / ((rect1_landw[0] + rect2_landw[0]) / 2) > 0.01 )
        return 0;
    if( (rect1_angle - rect2_angle) / ((rect1_angle + rect2_angle) / 2) > 0.1 )
        return 0;
    
    return 1;
}


std::vector<Point2f> combine_rect(RotatedRect rect1, RotatedRect rect2, Mat frame)
{
    float rect1_center[2] = {rect1.center.x , rect1.center.y};
    float rect2_center[2] = {rect2.center.x , rect2.center.y};  // 记录矩形中心

    float rect1_landw[2] = {0,0};
    float rect2_landw[2] = {0,0};
    find_length_and_width(rect1.size.width, rect1.size.height, rect1_landw);
    find_length_and_width(rect2.size.width, rect2.size.height, rect2_landw);  // 寻找矩形长宽
    
    float x_diff = rect2_center[0] - rect1_center[0];
    float y_diff = rect2_center[1] - rect1_center[1];
    
    float avg_length_pre = (rect1_landw[0] + rect2_landw[0])/ 2;
    float m1 = 0.9;
    float avg_length = avg_length_pre - (rect1_landw[1] + rect2_landw[1]) / 2;
    //消去灯条宽度带来的length误差
    
    float m2 = 0.9;
    float avg_width = (std::sqrt(std::pow(rect2_center[0] - rect1_center[0], 2) + std::pow(rect2_center[1] - rect1_center[1], 2)));
    // 消去width的sqrt误差的系数
    
    float avg_dia = std::sqrt(std::pow(avg_length, 2) + std::pow(avg_width, 2));
    float k1 = 0;
    if(x_diff != 0)
        k1 = y_diff / x_diff;  // 整体倾斜角，右斜为正
    float k2 = avg_length / avg_width;  // 装甲板矩形倾斜角
    Point2f center((rect1_center[0] + rect2_center[0])/2, (rect1_center[1]+rect2_center[1])/2);

    circle(frame, center, 3, Scalar(255,0,120),-1);
    //imshow("center",frame);

    return rect_point_cal(avg_dia, avg_width, k1, k2, center);
}

//___________________________________轮廓检测主函数__________________________________

std::vector<std::vector<Point2f>> direct_rect(std::vector<std::vector<cv::Point> > Point_fix, Mat frame){
    
    
    std::vector<RotatedRect> minAreaRects;
    RotatedRect temp;

    //-------找到小矩形轮廓-------
    for (int i = 0; i < Point_fix.size(); i++)
    {
        temp = minAreaRect(Point_fix[i]);
            if(temp.size.width * temp.size.height > 100)
                minAreaRects.push_back(minAreaRect(Point_fix[i]));
    }

    //-------排序-------
    sort(minAreaRects.begin(), minAreaRects.end(), compare_Left_or_Right);

    //-------调试矩形稳定程度-------
    Point2f rect_point[4];
    for(int k = 0; k < 2; k++)
    {   
        minAreaRects[k].points(rect_point);
        draw_rect_array(rect_point, frame);
    }
 
    //-------合成-------
    std::vector<std::vector<Point2f>> manyrect;
    for(int j = 0; j < (minAreaRects.size() - 1); j++)
    {
        manyrect.push_back( combine_rect(minAreaRects[j], minAreaRects[j+1], frame));
    }
    return manyrect;

}

//-------------------------------------------主函数-----------------------------------------------------

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
        std::vector<cv::Point3d> world_point;  // 用于PnP输入世界坐标系点集
        std::vector<cv::Point2d> img_point;  // 用于PnP输入像素坐标点集

        float last_rect_angle[10] = {0};  // 储存上一帧角度信息

        //-------开始读取-------
        while(cap.read(frame))
        {   
             //-------测距参数------
            double DISTANCE_DEEP_ZERO = 0;  // 用于记录第一帧（假设深度为零）的中心点image距离
            double deep_cal = 0;  // 用于记录深度
            
            
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
                drawContours(frame, Point_fix, -1, Scalar(255, 0, 255), -1);
                imshow("drawframe",frame);
                //-------轮廓、中心点检测（主要模块1)-------
                std::vector<std::vector<Point2f>> Rect_angpoint;
                Rect_angpoint = direct_rect(Point_fix, frame);
                
                for(int i = 0; i < Rect_angpoint.size(); i++)
                    draw_rect(Rect_angpoint[i], frame);
                imshow("Rect",frame);
                //draw_point(frame, center_point, rect_centerpoint);

                //distances_detect(minAreaRects, center_point, rect_centerpoint, No_rect, frame);

            }
        if(cv::waitKey(100) >= 0)  //一秒约24帧,按下键盘任意键退出
            break;
        }
    }
    return 0;
}


