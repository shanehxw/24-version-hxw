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

//---------------------------------------装甲板组------------------------------------------------------------

class armour{
    public:
        RotatedRect left;
        RotatedRect right;
        float angle;
        float length;
        float width;
    private:
        int id;
}














//----------------------------------------预参数-----------------------------------------------------

const Mat CAMERA_MADRIX = ( Mat_<double>(3,3) << 1400,0,360,0,1400,240,0,0,1);  // 相机内参
const Mat DIST_COEFFS = ( Mat_<double>(1,5) << 0, 0, 0, 0, 0 );  // 畸变矩阵

const double ARMOUR_LENGTH = 200;
const double ARMOUR_WIDTH = 100;  // 装甲板长宽

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

//-------------------------------------轮廓、中心点检测1.0（主要模块1)----------------------------------------------------

std::vector<RotatedRect> direct_centerpoint(Mat dst, double center_point[], double rect_centerpoint [], int No_rect[], std::vector<std::vector<cv::Point> > Point_fix){
    
    //-------参数-------
    int i_temp = 0;
    int j_temp = 0;  
    int k_temp = 0;

    //-------画出重要点-------
    std::vector<RotatedRect> minAreaRects(Point_fix.size());
    double k = 0;  // 第二个点和第四个点的斜率，用于世界坐标系赋值

    //-------遍历所有轮廓-------
    for (i_temp = 0 , j_temp = 0, k_temp = 0; i_temp < Point_fix.size(); i_temp++)
    {   
		minAreaRects[i_temp] = minAreaRect(Point_fix[i_temp]);  // 获取矩形
		if(minAreaRects[i_temp].size.width * minAreaRects[i_temp].size.height >= 50)  // 过滤面积较小的轮廓
        {   
            //-------记录矩形中点的x，y坐标-------
            rect_centerpoint[j_temp] = minAreaRects[i_temp].center.x;
            rect_centerpoint[j_temp+1] += minAreaRects[i_temp].center.y;
            //std::cout<<rect_centerpoint[j_temp]<<" "<<rect_centerpoint[j_temp+1]<<std::endl;
            j_temp += 2;  // 记录矩形中点的x，y坐标

            //-------记录矩形序号-------
            No_rect [k_temp]= i_temp;
            k_temp++;

            //-------调试中心点-------
            //circle(frame, minAreaRects[i_temp].center, 10, Scalar(255, 0, 120), -1);
            //std::cout<<"x = "<< minAreaRects[i].center.x << " y = " <<minAreaRects[i].center.y <<std::endl;
            //std::cout<<"sum_x = "<<sum_x<<" sum_y = "<<sum_y<<std::endl;
            }
    }

    center_point[0] = double ((rect_centerpoint[0] + rect_centerpoint[2]) / 2);
    center_point[1] = double ((rect_centerpoint[1] + rect_centerpoint[3]) / 2);   

    return minAreaRects;

}

void draw_point(Mat frame, double center_point[], double rect_centerpoint []){

    if(rect_centerpoint[0] && rect_centerpoint[1] && rect_centerpoint[2] && rect_centerpoint[3])
    {   
        Point2d rect_point1 (rect_centerpoint[0], rect_centerpoint[1]);
        Point2d rect_point2 (rect_centerpoint[2], rect_centerpoint[3]);
        Point2d center_target(center_point[0], center_point[1]);

        circle(frame, rect_point1, 10, Scalar(255, 0, 120), -1);
        circle(frame, rect_point2, 10, Scalar(255, 0, 120), -1);
        circle(frame, center_target, 10, Scalar(255, 0, 120), -1);
        
        imshow("point",frame);
    }

}

//------------------------------------------轮廓检索1.1------------------------------------------------------------

bool compare_Left_or_Right(const RotatedRect& pt1, const RotatedRect& pt2)
{
	if (pt1.center.x != pt2.center.x)
		return pt1.center.x < pt2.center.x;  // x从小到大排序
	else
		return pt1.center.y < pt2.center.y;
}



cv::Point2f mid_point(Point2f a, Point2f b)
{
    Point2f mid (a.x + b.x , a.y + b.y);
    return mid;
}

std::vector<RotatedRect> pointrank1(std::vector<RotatedRect> Rect_point)
{  

        std::sort(Rect_point.begin(), Rect_point.end(), compare_Left_or_Right);

        return Rect_point;
}

bool rect_compare(RotatedRect rect1, RotatedRect rect2)
{   
    float rect1_length = rect1.size.width > rect1.size.height ? rect1.size.width : rect1.size.height;
    float rect1_width = rect1.size.width <= rect1.size.height ? rect1.size.width : rect1.size.height;
    float rect2_length = rect2.size.width > rect2.size.height ? rect2.size.width : rect2.size.height;
    float rect2_width = rect2.size.width <= rect2.size.height ? rect2.size.width : rect2.size.height;

    if( (rect1_length-rect2_length) / ((rect1_length + rect2_length) / 2) > 0.1 )
        return false;
    if( (rect1.angle - rect2.angle) / ((rect1.angle + rect2.angle) / 2) >0.1 )
        return false;
    
    return true;
}



std::vector<Point2f> rect_point_cal(float avg_dia, float k1_world, float k2_armour, Point2f center)
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

    // 点的方案
    Point2f point_RIGHTDOWM (center.x + ang_to_center_x_1, center.y + ang_to_center_y_1);  // 求出右下点
    Point2f point_LEFTDOWN (center.x - ang_to_center_x_2, center.y + ang_to_center_y_2);  // 求出左下点
    Point2f point_LEFTUP (center.x - ang_to_center_x_1, center.y - ang_to_center_y_1);  // 求出左上点
    Point2f point_RIGHTUP (center.x + ang_to_center_x_2, center.y - ang_to_center_y_2);  // 求出右上点
    
    std::vector<cv::Point2f> rect_combine;
    rect_combine.push_back(point_RIGHTDOWM);
    rect_combine.push_back(point_LEFTDOWN);
    rect_combine.push_back(point_LEFTUP);
    rect_combine.push_back(point_RIGHTUP);  // 右下-左下-左上-右上

    return rect_combine;
}

std::vector<Point2f> combine_rect(RotatedRect rect1, RotatedRect rect2, Mat frame)
{

   
    float x_diff = rect2.center.x - rect1.center.x;
    float y_diff = rect2.center.y - rect1.center.y;
    float avg_length_pre = ((rect1.size.width > rect1.size.height ? rect1.size.width : rect1.size.height) + (rect2.size.width > rect2.size.height ? rect2.size.width : rect2.size.height))/2;
    float avg_length = ((avg_length_pre - ((rect1.size.width < rect1.size.height ? rect1.size.width : rect1.size.height) + (rect2.size.width < rect2.size.height ? rect2.size.width : rect1.size.height)) / 2) / avg_length_pre) * avg_length_pre;
    //消去灯条宽度带来的length误差
    float m2 =0.5;
    float avg_width = m2*(std::sqrt(std::pow(rect2.center.x - rect1.center.x, 2) + std::pow(rect2.center.y - rect1.center.y, 2)));
    // 消去width的sqrt误差的系数
    float avg_dia = std::sqrt(std::pow(avg_length, 2) + std::pow(avg_width, 2));
    float k1 = 0;
    if(x_diff != 0)
        k1 = y_diff / x_diff;  // 整体倾斜角，右斜为正
    float k2 = avg_length / avg_width;  // 装甲板矩形倾斜角
    Point2f center((rect2.center.x + rect1.center.x)/2, (rect2.center.y+rect1.center.y)/2);

    circle(frame, center, 3, Scalar(255,0,120),-1);
    imshow("center",frame);

    return rect_point_cal(avg_dia, k1, k2, center);
}



void draw_rect(std::vector<Point2f> rect, Mat frame)
{   
    for(int i = 0; i < rect.size(); i++)
    {   
        line(frame, rect[i%rect.size()], rect[(i+1)%rect.size()], Scalar(255,0,120), 3);
    }
}



std::vector<std::vector<Point2f>> direct_rect(std::vector<std::vector<cv::Point> > Point_fix, Mat frame){
    
    
    std::vector<RotatedRect> minAreaRects;
    RotatedRect temp;
    //-------找到小矩形轮廓-------
    for (int i = 0; i < Point_fix.size(); i++)
    {   
        temp = minAreaRect(Point_fix[i]);
        if()
		minAreaRects.push_back(temp);
    }

    //-------排序-------
    sort(minAreaRects.begin(), minAreaRects.end(), compare_Left_or_Right);

    //-------合成-------
    std::vector<std::vector<Point2f>> manyrect;
    for(int j = 0; j < (minAreaRects.size() - 1); j++)
    {
        manyrect.push_back( combine_rect(minAreaRects[j], minAreaRects[j+1], frame));
    }
    return manyrect;

}


//------------------------------------------点的排序-----------------------------------------------------

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
                cv::imshow("ori",frame);
                
                Mat baw = pretreatment(frame);
                
                Mat mask = region_interested(baw, CAP_HEIGHT, CAP_WIDTH);

                Mat dst = filtering(mask);
                
                //-------检索外部轮廓-------
                std::vector<std::vector<cv::Point> > Point_fix;
                std::vector<cv::Vec4i> hierarchy;
                findContours(dst, Point_fix, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
                
                //-------轮廓、中心点检测（主要模块1)-------
                std::vector<std::vector<Point2f>> Rect_angpoint;
                Rect_angpoint = direct_rect(Point_fix, frame);
                
                for(int i = 0; i < Rect_angpoint.size(); i++)
                    draw_rect(Rect_angpoint[i], frame);
                imshow("Rect",frame);
                //draw_point(frame, center_point, rect_centerpoint);

                //distances_detect(minAreaRects, center_point, rect_centerpoint, No_rect, frame);
                
            }
        if(cv::waitKey(50) >= 0)  //一秒约24帧,按下键盘任意键退出
            break;
        }
    }
    return 0;
}


