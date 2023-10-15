#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp> 

using namespace cv;

//--------------------------------------------------------------------------------------------------------------

const double PI = 3.14159265;

//---------------------------------------------------------工具库------------------------------------------------

void find_width_and_height(float width, float height, float* array)  // 对minareaRect的宽高进行排序:大的在前
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

void draw_rect_array(Point2f rect_point[], Mat frame, Scalar color)  // 通过Point2f[4]画出矩形
{ 
    for(int i = 0; i < 4; i++)
    {  
        line(frame, rect_point[i%4], rect_point[(i+1)%4], color, 2);
    }
}

void draw_rect(std::vector<Point2f> rect, Mat frame, Scalar color)  // 通过vector<Point2f>画出矩形
{   
    for(int i = 0; i < 4; i++)
    {   
        line(frame, rect[i%4], rect[(i+1)%4], color, 1);
    }
}

float calculateDistance(cv::Point2f pt1, cv::Point2f pt2)   // 通过两个Point2f计算距离
{  

    float dx = pt2.x - pt1.x;  

    float dy = pt2.y - pt1.y;  

    return std::sqrt(dx * dx + dy * dy);  

} 

//---------------------------------------------------------二值化--------------------------------------------------------------
Mat img_gray(Mat frame)  // 灰度图转化
{   
    Mat dst_gray;
    Mat dst_baw;
    cv::cvtColor(frame, dst_gray, COLOR_BGR2GRAY);
    //imshow("gray",dst_gray);
    cv::threshold(dst_gray, dst_baw, 120, 255, cv::THRESH_BINARY);  // 不能调太狠，因为装甲板的灯条需要完全识别到
    //imshow("gray_baw",dst_baw);

    return dst_baw;

}

Mat img_channel(Mat frame)  // BGR通道处理
{
    std::vector<cv::Mat> bgr_channels;
                
    cv::split(frame, bgr_channels);

    //通道相减
    Mat imageChannelcal = bgr_channels[0];

    //-------高斯低通滤波------
    Mat frame_gaublur;
    cv::GaussianBlur(imageChannelcal, frame_gaublur, Size(3,3), 0, 0);  // 高斯滤波处理
    

    //-------二值化-------
    cv::Mat dst_baw;
    cv::threshold(imageChannelcal, dst_baw, 40, 255, cv::THRESH_BINARY);
    //imshow("channel",dst_baw);

    return dst_baw;
}

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


//----------------------------------------------------------滤波处理----------------------------------------------------------------------

Mat filter(Mat dst){

    //cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));  //池化核大小
    //cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));  //池化核大小
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    
    //cv::erode(dst, dst, kernel7, Point(-1,-1), 1);
    cv::dilate(dst, dst, kernel3, Point(-1,-1), 1);
    
    cv::GaussianBlur(dst, dst, Size(5,5), 0, 0);  // 高斯滤波处理

    //blur(dst, dst, Size (5,5));
    imshow("filter", dst);
    return dst;
}

//-------------------------------------------------------矩形检测1------------------------------------------------------------------------
//长宽比筛选 -> 角度初步筛选（因为装甲板实际上大部分都是垂直于地面，可以筛一些离谱噪声）
std::vector<RotatedRect> rect_detect1(Mat dst, Mat img)
{
    //-------提取轮廓-------
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        //debug
        drawContours(img, contours, -1, Scalar(255, 0, 0), -1);
        //imshow("drawframe",img);

    //-------minAreaRect + 长宽筛选 + 矩形本身斜率筛选-------
    std::vector<RotatedRect> rect_pre1;  // 长宽比检测得到的矩形轮廓pre1
    std::vector<float*> rect_pre1_wah;  // 储存pre1长宽
    //std::vector<float> rect_pre1_longer_angle;  // 储存pre1较长边与水平夹角，使用atan，范围在-90～90、
                                                // 每个矩形的角度暂存在angle_temp里

    float k_wah_min = 2;
    float k_wah_max = 5;  //设置的长宽比阈值
    float angle_border = 60; //设置的单个矩形的角度阈值
    for (int i = 0; i < contours.size(); i++)
    {
        RotatedRect temp = minAreaRect(contours[i]);  // minAreaRect
        float temp_wah[2] = {0};
        Point2f temp_points[4];
        find_width_and_height(temp.size.width, temp.size.height, temp_wah);
            if( k_wah_min < (temp_wah[0] / temp_wah[1]) && (temp_wah[0] / temp_wah[1]) < k_wah_max ){  // 长宽筛选结束
                    //--debug
                    Point2f* point_debug;
                // 计算该矩形较长边斜率
                temp.points(temp_points);
                for(int j = 0; j < 4; j++){
                    float dis = calculateDistance(temp_points[j%4], temp_points[(j+1)%4]);
                    if(temp_wah[0] - 1 <= dis && dis <= temp_wah[0] + 1){  // 找到较长边对应的矩形
                        float angle_temp;
                        if(std::abs(temp_points[j%4].x - temp_points[(j+1)%4].x) < 1e-6){  // 防止可能的截断
                            angle_temp = 90; 
                        }
                        else{
                            float k_temp = (temp_points[j%4].y - temp_points[(j+1)%4].y) / (temp_points[j%4].x - temp_points[(j+1)%4].x);
                            angle_temp = std::atan(k_temp) * 180 / PI;
                        }
                            //--debug
                            //std::cout<<std::abs(temp_points[j%4].x - temp_points[(j+1)%4].x)<<"-------"<<angle_temp<<std::endl;
                        if(angle_temp < (-angle_border) || angle_border < angle_temp){
                            rect_pre1.push_back(temp);
                                //--debug
                                point_debug = temp_points;
                                draw_rect_array(point_debug, img, Scalar(0,255,255));
                        }
                    }
                }
                    
            }
    }
        //debug
        imshow("pre1",img);
        
    //-------根据位置分组，通过面积计算出当前矩形的roi范围，然后在范围里分组*，筛去两个中心点所连成的直线的斜率较大的/我新写一个函数，计算rect较长边的斜率-------
    float k_roi_pre1 = ;  // 假设面积的平方根和roi范围是正比关系，目前来看是一个经验值，但是应该可以通过计算得到更精确值
    std::vector<float> roi_pre1;
    std::vector<Point2f> center_pre1;
    std::vector<float> k_rect;  // 记录所需要的值：roi和中心点
    for(int i = 0; i < rect_pre1.size(); i++){
        //roi
        float temp_area = rect_pre1_wah[i][0] * rect_pre1_wah[i][1];
        float temp_roi = std::sqrt(temp_area) * k_roi_pre1;
        roi_pre1.push_back(temp_roi);
        //中心点
        center_pre1.push_back(rect_pre1.center);
    }

    //考虑分组器受噪音干扰较强，最好还要再先判断一次角度：两个中心点连线的斜率
    //分组器：1、点i是否是第一次拉人进组，决定是拉一个点还是拉两个点（最后考虑） 2、点i拉的点j是否之前已经有组，决定是否要将组合并 3、点i是否之前已经有组，再拉点j入组的时候需要对组进行扩充 4、点i的组和点j的组是否是同一个组
    //因此有：点与点、点与组、组与组的合并，并且在for循环完之后，要确认是否组号是否有间断，要进行补偿
    //组号和进组的顺序弱相关 --改进方案三解决
    //改进方案1：直接使用栈控制组号，提前初始化好栈，当组号空出来时压栈
    //改进方案2：可以使用类去管理（待改进）：一个点的两个属性：数据和组号
    //改进方案3：直接用vector去存储组里面点的序号，好处是点的序号和点的数据信息在vector位置中是同步的，并且不用多余的栈去管理组号，缺点是判断点是否有组时，需要遍历
    //         使用数组：方便确定点的组，但是会造成数据位置和组号位置的错位； 使用vector：方便确定组号和组的点（组的点无需确定，因为只关心数据本身），但是每次合并组、删除组、判断点的组的时候会遍历
    //--还是决定还是使用类进行分组
    float angle_center_border = 50;  // 中心点间斜率
    std::vector<std::vector<Point2f>> armour_group;  // 用于储存结果
    int i_pre2 = 0;
    int j_pre2 = 0;
    int if_group_pre2 = 0;  // 判断是否有缓存组，确定是拉一个点还是两个点
    //int if_joined[center_pre1.size()] = {0};  // 放入组号，让点和组号一一对应
    std::vector<int> if_joined;  // 用vector管理点的组号
    std::deque<int> group_num;  // 用栈管理组号的分配，用于放入if_joined
    for(int i = center_pre1.size(); i > 0; i--){  // 栈的初始化
        group_num.push(i);
    }
    if(!group_num.empty()){
        for(i_pre2 = 0; i_pre2 < center_pre1.size(); i_pre2++){
            if_group_pre2 = 0;
            std::vector<Point2f> temp_group;  // 点i的缓存组
            for(j_pre2 = 0; j_pre2 < center_pre1.size(); j_pre2++){
                //计算中心点斜率并排除一些无关点
                float k_center;
                float angle_center;
                if( std::abs(center_pre1[i].x - center_pre1[j].x) < 1e-6){
                    continue;
                }
                else{
                    k_center = (center_pre1[i].y - center_pre1[j].y) / (center_pre1[i].x - center_pre1[j].x);
                    angle_center = std::atan(k_center) * 180 / PI;
                    if(angle_center_border < angle_center || angle_center < -angle_center_border){
                        continue;
                    }
                }
                //开始分组
                float center_dis = calculateDistance(center_pre1[i], center_pre1[j]);
                if(center_dis < roi_pre1[i]){  // 判断点j是否在点i的相关范围内，如果参数取的好，每组点可以进行两次判断然后取交集
                    //首先假设点i的组不是点j的组
                    //if(if_joined[i] > 0 && if_joined[j] > 0){  // 两个都有组，把j的组加进i的组，j的组号空了出来，把j的组号压入栈
                                                // 3：两个都有组，把j的组加进i的组，把j的序号加入到i的组号中，删除vector中j的组，和vector中j的组号
                                                // --改进：把j的组整体加入缓存区的组，缓存区需要添加一个序号的缓存，（缓存区指的是这个循环的缓存区，暂时不对外部vector更新）
                                                //关键是把组和组号都移动，并且
                    //if(if_joined[i] > 0 && if_joined[j] == 0){  // i有组，把j加进i的组，更新j的组号，组号计数器不增加
                                                // 3：i有组，把j加进i的组，把j的序号加进i的组号 --改进：把j放入缓存组中
                    
                    //if(if_joined[i] == 0 && if_joined[j] > 0){  // j有组，把i在缓存区的组加进j的组，并且更新i组里所有元素的组号，组号计数器不变
                                                // 3：在该分支里创建一个组号
                    
                    //if(if_joined[i] == 0 && if_joined[j] == 0){  //i、j没组，新增一个组进入缓存区，到末尾从栈中取出一个组号
                
                    if(if_joined[i] > 0){  // i有组
                        if(if_joined[j] > 0){  // 两个都有组，把j的组加进i的组，j的组号空了出来，把j的组号压入栈
                            1
                        }
                    }
                
                
                    if_group_pre2++;
                    if(if_group_pre2 == 1){
                        if()
                        temp_group.push_back(center_pre1[i]);
                        temp_group.push_back(center_pre1[j]);
                        if_joined[i] = group_num;
                        if_joined[j] = group_num;
                    }
                    else if(if_group_pre2 >= 1 && ){
                        temp_group.push_back(center_pre1[j]);
                        if_joined[j] = group_num;
                    }
                group_num++;
                }
            }
            temp_group.push_back(temp_group);  // group_num-1是对应temp_group的序号
        
        }
    }
    return rect_pre1;
}

//-----------------------------------------------------------------------------------------------------------------------------------------

int main(){
    Mat img = imread("../armour_blue.png");
    if(img.data != nullptr){
        Mat baw = img_baw(img);
        baw =filter(baw);
        rect_detect1(baw, img);
    }

    waitKey(0);  

    return 0;
}