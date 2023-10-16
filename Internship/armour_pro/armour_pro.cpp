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

cv::Point2f mid_point(Point2f a, Point2f b)  // 求Point中心点
{
    Point2f mid (a.x + b.x , a.y + b.y);
    return mid;
}


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

//---------------------------------------------------------排序------------------------------------------------------------------------

bool compare_Left_or_Right(const RotatedRect& pt1, const RotatedRect& pt2)  // 通过中心点对矩形进行从左到右的排序
{
	return pt1.center.x < pt2.center.x;  // x从小到大排序
}

std::vector<RotatedRect> pointrank1(std::vector<RotatedRect> Rect_point)  // compare_Left_or_Right的触发函数
{  
    std::sort(Rect_point.begin(), Rect_point.end(), compare_Left_or_Right);
    return Rect_point;
}

//---------------------------------------------------------并查集-------------------------------------------------------------------

void init(int array[], int array_size){  // 初始化并查集，每个节点都是它本身
    for(int i = 0; i < array_size; i++){
        array[i] = i;
    }
}

int find(int array[], int x){  // 查找根,对array里的x查找
    if(array[x] == x){
        return x;
    }
    return array[x] = find(array,array[x]);  // 路径压缩优化
}

void merge(int array[], int array_size, int x, int y){  // 并查集的合并--判断点的根是否相同，如果不同，对较大的根序号进行搜索，将所有较大根序号，改为较小的根序号
    int root_x = find(array, x);
    int root_y = find(array, y);
    if(root_x > root_y){
        for(int i = 0; i < array_size; i++){
            if(root_x == find(array,i)){
                array[i] = root_y;
            }
        }
    }
    else if(root_x < root_y){
        for(int i = 0; i < array_size; i++){
            if(root_y == find(array,i)){
                array[i] = root_x;
            }
        }
    }

}


//---------------------------------------------------------二值化--------------------------------------------------------------
Mat img_gray(Mat frame)  // 灰度图转化
{   
    Mat dst_gray;
    Mat dst_baw;
    cv::cvtColor(frame, dst_gray, COLOR_BGR2GRAY);
    //imshow("gray",dst_gray);
    cv::threshold(dst_gray, dst_gray, 120, 255, cv::THRESH_BINARY);  // 不能调太狠，因为装甲板的灯条需要完全识别到
    //imshow("dst_gray",dst_gray);

    return dst_gray;

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
    cv::Mat dst_baw_channel;
    cv::threshold(imageChannelcal, dst_baw_channel, 40, 255, cv::THRESH_BINARY);
    //imshow("channel",dst_baw_channel);

    return dst_baw_channel;
}

//-------通道分离1,灯条识别-------
Mat img_baw_light(Mat frame)
{   
    // 灰度图处理
    Mat dst_gray;
    Mat dst_baw;
    cv::cvtColor(frame, dst_gray, COLOR_BGR2GRAY);
    //imshow("gray",dst_gray);
    cv::threshold(dst_gray, dst_gray, 120, 255, cv::THRESH_BINARY);  // 不能调太狠，因为装甲板的灯条需要完全识别到
    //imshow("dst_gray",dst_gray);

    // BGR处理
    std::vector<cv::Mat> bgr_channels;
    cv::split(frame, bgr_channels);
    //通道相减
    Mat imageChannelcal = bgr_channels[0];
    // 高斯滤波处理
    Mat frame_gaublur;
    cv::GaussianBlur(imageChannelcal, frame_gaublur, Size(3,3), 0, 0);
    //二值化
    cv::Mat dst_baw_channel;
    cv::threshold(imageChannelcal, dst_baw_channel, 40, 255, cv::THRESH_BINARY);
    //imshow("channel",dst_baw_channel);

    //对两个二值化图像进行交运算,降低噪声
    Mat img_and;
    bitwise_and(dst_gray, dst_baw_channel, img_and);
    //imshow("img_and",img_and);
    
    return img_and;
}

//-------通道分离1,数字识别-------
Mat img_baw_number(Mat frame)
{
    // BGR处理
    std::vector<cv::Mat> bgr_channels;
    cv::split(frame, bgr_channels);
    //通道相减
    Mat imageChannelcal = bgr_channels[1] + bgr_channels[2] -  bgr_channels[0] ;
    // 高斯滤波处理
    //Mat frame_gaublur;
    //cv::GaussianBlur(imageChannelcal, frame_gaublur, Size(3,3), 0, 0);
    //二值化
    cv::Mat dst_baw_channel;
    cv::threshold(imageChannelcal, dst_baw_channel, 2, 20, cv::THRESH_BINARY);
    imshow("channel",dst_baw_channel);
    
    return dst_baw_channel;

}

//----------------------------------------------------------滤波处理----------------------------------------------------------------------

Mat filter_light(Mat dst)
{

    //cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));  //池化核大小
    //cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));  //池化核大小
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    
    //cv::erode(dst, dst, kernel7, Point(-1,-1), 1);
    cv::dilate(dst, dst, kernel3, Point(-1,-1), 1);
    
    cv::GaussianBlur(dst, dst, Size(5,5), 0, 0);  // 高斯滤波处理

    //blur(dst, dst, Size (5,5));
    //imshow("filter", dst);
    return dst;
}

Mat filter_number(Mat dst)
{
    //cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));  //池化核大小
    //cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));  //池化核大小
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    
    //cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    cv::dilate(dst, dst, kernel3, Point(-1,-1), 3);
    
    cv::GaussianBlur(dst, dst, Size(5,5), 0, 0); 
    cv::GaussianBlur(dst, dst, Size(5,5), 0, 0);  // 高斯滤波处理

    //blur(dst, dst, Size (5,5));
    imshow("filter", dst);
    return dst;
}


//----------------------------------------------------------灯条检测1------------------------------------------------------------------------
//长宽比筛选 -> 角度初步筛选（因为装甲板实际上大部分都是垂直于地面，可以筛一些离谱噪声）
std::vector<std::vector<RotatedRect>> rect_detect1(Mat dst, Mat img)
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
                            rect_pre1_wah.push_back(temp_wah);
                                //--debug
                                point_debug = temp_points;
                                draw_rect_array(point_debug, img, Scalar(0,255,255));
                            break;
                        }
                    }
                }
                    
            }
    }
        //--debug
        //imshow("pre1",img);
        
    //-------根据位置分组，通过面积计算出当前矩形的roi范围，然后在范围里分组*，筛去两个中心点所连成的直线的斜率较大的/我新写一个函数，计算rect较长边的斜率-------
    float k_roimin_pre1 = 2;
    float k_roimax_pre1 = 6;  // 假设面积的平方根和roi范围是正比关系，目前来看是一个经验值，但是应该可以通过计算得到更精确值
    std::vector<float> roimin_pre1;
    std::vector<float> roimax_pre1;  // 感兴趣区域半径
    std::vector<Point2f> center_pre1;
    std::vector<float> k_rect;  // 记录所需要的值：roi和中心点
    for(int i = 0; i < rect_pre1.size(); i++){  // 注意，center序号和pre1序号一致
        //roi
        float temp_area = rect_pre1_wah[i][0] * rect_pre1_wah[i][1];
        float temp_roimin = std::sqrt(temp_area) * k_roimin_pre1;
        float temp_roimax = std::sqrt(temp_area) * k_roimax_pre1;
        roimin_pre1.push_back(temp_roimin);
        roimax_pre1.push_back(temp_roimax);
        //中心点
        center_pre1.push_back(rect_pre1[i].center);
    }
        //--debug
        std::cout<<"center_pre1 = "<<center_pre1.size()<<std::endl;

    //考虑分组器受噪音干扰较强，最好还要再先判断一次角度：两个中心点连线的斜率
    //改进方案4：使用数组实现并查集控制组的序号
    float angle_center_border = 40;  // 中心点间斜率
    std::vector<std::vector<RotatedRect>> armour_group;  // 用于储存结果
    int i_pre2 = 0;
    int j_pre2 = 0;
    int pre1_size = center_pre1.size();
    int group_size[pre1_size] = {0};  // 储存每个组有几个矩形，初始值是零
    int group_num[pre1_size] = {0};  // 并查集数组
    init(group_num, pre1_size);  // 初始化并查集
    
        if(group_num[1] == 1){  // 确定并查集初始化没问题
        for(i_pre2 = 0; i_pre2 < pre1_size; i_pre2++){
            std::vector<Point2f> temp_group;  // 点i的缓存组
            for(j_pre2 = 0; j_pre2 < pre1_size; j_pre2++){
                //计算中心点斜率并排除一些无关点
                float k_center;
                float angle_center;
                if( std::abs(center_pre1[i_pre2].x - center_pre1[j_pre2].x) < 1e-6){
                    continue;
                }
                else{
                    k_center = (center_pre1[i_pre2].y - center_pre1[j_pre2].y) / (center_pre1[i_pre2].x - center_pre1[j_pre2].x);
                    angle_center = std::atan(k_center) * 180 / PI;
                    if(angle_center_border < angle_center || angle_center < -angle_center_border){
                        continue;
                    }
                }
                //开始分组,关键是对group_num进行操作
                float center_dis = calculateDistance(center_pre1[i_pre2], center_pre1[j_pre2]);
                if(roimin_pre1[i_pre2] < center_dis && center_dis < roimax_pre1[i_pre2]){  // 判断点j是否在点i的相关范围内，如果参数取的好，每组点可以进行两次判断然后取交集
                    merge(group_num, pre1_size, i_pre2, j_pre2);  // 两个组合并
                }
            }
        }
        for(i_pre2 = 0; i_pre2 < pre1_size; i_pre2++){  // 计算各组的矩形个数
            group_size[find(group_num, i_pre2)]++;
        }
        for(i_pre2 = 0; i_pre2 < pre1_size; i_pre2++){
                //--debug
                //std::cout<<"group_size = "<<group_size[i_pre2]<<std::endl;
            if(group_size[i_pre2] < 5 && group_size[i_pre2] > 1){  // 如果当前组的矩形个数在2-4之间
                std::vector<RotatedRect> temp;
                for(j_pre2 = 0; j_pre2 < pre1_size; j_pre2++){
                    if(i_pre2 == find(group_num, j_pre2)){  //搜索所有指向i_pre2的元素
                        temp.push_back(rect_pre1[j_pre2]);
                    }
                }
                armour_group.push_back(temp);
            }
        }
    }  // 分组结束

        //--debug
        for(int i = 0; i < armour_group.size(); i++){
            std::vector<RotatedRect> temp_group;
            temp_group = armour_group[i];
           for(int j = 0; j < temp_group.size(); j++){
                circle(img, temp_group[j].center, 10, Scalar(255,255,255), -1);
            }
        }
        //--debug
        std::cout<<"armour_group.size() = "<<armour_group.size()<<std::endl;
        imshow("group", img);

    // 每个组里的矩形从左到右排序，并且计算距离
    std::vector<std::vector<RotatedRect>> armour_group_inorder;
    for(int i = 0; i < armour_group.size(); i++){
        std::vector<RotatedRect> temp = pointrank1(armour_group[i]);
        armour_group_inorder.push_back(temp);
    }

    
    
    return armour_group_inorder;
}

//----------------------------------------------------感兴趣区域提取、处理--------------------------------------------------------------------------

std::vector<std::vector<Point2f>> Roical_for_num(std::vector<RotatedRect> group)  // 第i个Roi关键点对应group里面的第i个和第i+1个矩形
{   
    int size = group.size();
    std::vector<float*> rect_wah;  // 计算并储存长宽
    for(int i = 0; i < size; i++){
        float temp_wah[2] = {0};
        find_width_and_height(group[i].size.width, group[i].size.height, temp_wah);
        rect_wah.push_back(temp_wah);
    }

    std::vector<std::vector<Point2f>> roi_points;  // 区域从左到右，区域顶点顺序左上开始逆时针
    for(int i = 0; i < size - 1; i++){  // 计算每两个相邻点的roi区域
        float roi_x = (group[i+1].center.x - group[i].center.x) /2 - (rect_wah[i][1] + rect_wah[i+1][1]) /2;
        float roi_y = (rect_wah[i][0] + rect_wah[i+1][0]) /2;
        float mid_x = (group[i].center.x + group[i+1].center.x) /2;
        float mid_y = (group[i].center.y + group[i+1].center.y) /2;
        Point2f leftup (mid_x - roi_x, mid_y - roi_y);
        Point2f leftdown (mid_x - roi_x, mid_y + roi_y);
        Point2f rightdown (mid_x + roi_x, mid_y + roi_y);
        Point2f rightup (mid_x + roi_x, mid_y - roi_y);
        std::vector<Point2f> temp;
        temp.push_back(leftup);
        temp.push_back(leftdown);
        temp.push_back(rightdown);
        temp.push_back(rightup);
        roi_points.push_back(temp);
    }
    return roi_points;
}

Mat Roi_fetch(Mat dst, std::vector<Point2f> angle_points)  // 深拷贝roi
{   
    Mat submat = dst( Range(angle_points[0].y - 1, angle_points[1].y) , Range(angle_points[0].x -1, angle_points[2].x));
    Mat roi_img = submat;
    return roi_img;
}

Mat Roi_erode(Mat dst)
{   
    Mat output;
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    cv::erode(dst, output, kernel3, Point(-1,-1), 2);
    cv::GaussianBlur(output, output, Size(5,5), 0, 0);
    return output;
}

bool Roi_erode_findnum(Mat roi_erode)
{   
    bool if_num = false;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(roi_erode, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    float area_min = roi_erode.rows * roi_erode.cols * 4 / 7;
        //--debug
        //std::cout<<"area_min = "<<area_min<<std::endl;
    for (int i = 0; i < contours.size(); i++){
        RotatedRect temp = minAreaRect(contours[i]);
        float temp_area = temp.size.width * temp.size.height;
            //--debug
            //std::cout<<"temp_area = "<<temp_area<<std::endl;
        if(temp_area > area_min){
            if_num = true;
            break;
        }

    }

    return if_num;
}

Mat Roi_dilate(Mat dst)
{
    Mat output;
    //cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));  //池化核大小
    cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));  //池化核大小
    //cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    
    cv::dilate(dst, output, kernel5, Point(-1,-1), 1);
    
    cv::GaussianBlur(output, output, Size(5,5), 0, 0);  // 高斯滤波处理

    return output;
}

bool Roi_dilate_findnum(Mat roi_dilate)
{
    bool if_num = false;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(roi_dilate, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    float area_min = roi_dilate.rows * roi_dilate.cols * 2 / 3;
    float area_max = roi_dilate.rows * roi_dilate.cols * 8 / 9;
        //--debug
        //std::cout<<"area_min = "<<area_min<<std::endl;
    for (int i = 0; i < contours.size(); i++){
        RotatedRect temp = minAreaRect(contours[i]);
        float temp_area = temp.size.width * temp.size.height;
        if(temp_area > area_min && temp_area < area_max){
            if_num = true;
        }

    }

    return if_num;
}

//首先通过相邻两个灯条的距离等参数计算出大致的Roi范围，然后在专门的检测数字的二值化图像中截取Roi区域，
//然后分为两组方法处理并用面积比作为阈值，先腐蚀后膨胀找到每一组机器人的装甲板位置
std::vector<std::vector<RotatedRect>> target_find_inRoi(std::vector<std::vector<RotatedRect>> armour_group, Mat dst_number, Mat img)
{
    int i = 0;
    int j = 0;
    std::vector<std::vector<RotatedRect>> target_group;
    for(i = 0; i < armour_group.size(); i++){
        std::vector<std::vector<Point2f>> Roi_points_fix = Roical_for_num(armour_group[i]);  // 存放i组里面所有的Roi关键点
        bool if_over = false;  // 控制：先腐蚀检测一遍，筛选完再膨胀
            //--debug
            for(j = 0; j < Roi_points_fix.size(); j++){
                draw_rect(Roi_points_fix[j], img, Scalar(255, 255, 255));
            }     
        //腐蚀检测
        for(j = 0; j < Roi_points_fix.size(); j++){
            Mat roi_img = Roi_fetch(dst_number, Roi_points_fix[j]);
            Mat roi_erode = Roi_erode(roi_img);
                //--debug
                if(i == 0 && j == 0){
                    imshow("5-0",roi_erode);
                }
                if(i == 0 && j == 1){
                    imshow("5-1",roi_erode);
                }
                if(i == 0 && j == 2){
                    imshow("5-2",roi_erode);
                }
                if(i == 1 && j == 0){
                    imshow("3-0",roi_erode);
                }
                if(i == 1 && j == 1){
                    imshow("3-1",roi_erode);
                }
                if(i == 1 && j == 2){
                    imshow("3-2",roi_erode);
                }
            if( Roi_erode_findnum(roi_erode) ){  // 可改进，没有设置检验，是否有两组或以上符合筛选条件的
                std::vector<RotatedRect> temp_for_find = armour_group[i];  // 原文件深拷贝
                std::vector<RotatedRect> target_tempforsave;  // 缓存区

                RotatedRect temp_for_save = temp_for_find[j];
                target_tempforsave.push_back(temp_for_save);
                temp_for_save = temp_for_find[j+1];
                target_tempforsave.push_back(temp_for_save);

                target_group.push_back(target_tempforsave);
                if_over = true;
                break;
            }  // 腐蚀筛选完成
        }

        if(if_over){
            continue;  // 跳过剩下的膨胀检测
        }
        // 膨胀检测
        for(j = 0; j < Roi_points_fix.size(); j++){ 
            Mat roi_img = Roi_fetch(dst_number, Roi_points_fix[j]);
            Mat roi_dilate = Roi_dilate(roi_img);
                if(i == 2 && j == 0){
                    imshow("1-0",roi_dilate);
                }
                if(i == 2 && j == 1){
                    imshow("1-1",roi_dilate);
                }
            if( Roi_dilate_findnum(roi_dilate) ){  // 可改进，没有设置检验，是否有两组或以上符合筛选条件的
                std::vector<RotatedRect> temp_for_find = armour_group[i];
                std::vector<RotatedRect> target_tempforsave;

                RotatedRect temp_for_save = temp_for_find[j];
                target_tempforsave.push_back(temp_for_save);
                temp_for_save = temp_for_find[j+1];
                target_tempforsave.push_back(temp_for_save);

                target_group.push_back(target_tempforsave);  // 从左到右存
                break;
            }  // 膨胀筛选完成
        }
    }

    return target_group;
}

//--------------------------------------------------------装甲板计算-------------------------------------------------------------------------------

std::vector<Point2f> rhom_point_cal(std::vector<RotatedRect> target)
{   
    float rect1_landw[2] = {0,0};
    float rect2_landw[2] = {0,0};
    find_width_and_height(target[0].size.width, target[0].size.height, rect1_landw);
    find_width_and_height(target[1].size.width, target[1].size.height, rect2_landw);  // 寻找矩形长宽

    float x_diff = target[1].center.x - target[0].center.x;
    float y_diff = target[1].center.y - target[0].center.y;

    float ar_width = (rect1_landw[0] + rect2_landw[0])/ 2;
    float ar_length = std::sqrt(x_diff * x_diff + y_diff * y_diff) - ((rect1_landw[1] + rect2_landw[1]) /2);

    
    Point2f leftpoint = target[0].center;
    Point2f rightpoint = target[1].center;
    Point2f mid ( (leftpoint.x + rightpoint.x) /2  , (leftpoint.y + rightpoint.y) /2);
    float center_x = mid.x;
    float center_y = mid.y;

    float goal_x_diff = ar_width * y_diff / ar_length;
    float goal_y_diff = ar_width * x_diff / ar_length;

    std::vector<Point2f> vertical_point;
    Point2f uppoint (center_x + goal_x_diff, center_y - goal_y_diff);
    Point2f downpoint (center_x - goal_x_diff, center_y + goal_y_diff);

    vertical_point.push_back(downpoint);
    vertical_point.push_back(leftpoint); 
    vertical_point.push_back(uppoint); 
    vertical_point.push_back(rightpoint);  //下、左、上、右
    vertical_point.push_back(mid);

    return vertical_point;

}

//-----------------------------------------------------------------------------------------------------------------------------------------

int main(){
    Mat img = imread("../armour_blue.png");
    if(img.data != nullptr){
        // 寻找灯条预处理
        Mat dst_light = img_baw_light(img);
        dst_light =filter_light(dst_light);
        // 寻找数字预处理
        Mat dst_number = img_baw_number(img);
        //dst_number = filter_number(dst_number);  // 可以先不处理，等到roi筛选后再处理
        //寻找灯条
        std::vector<std::vector<RotatedRect>> armour_group = rect_detect1(dst_light, img);
        //使用Roi检测数字得到目标
        std::vector<std::vector<RotatedRect>> target = target_find_inRoi(armour_group, dst_number, img);
            //--debug
            std::cout<<target.size()<<std::endl;

        std::vector<std::vector<Point2f>> target_armour_rioh;
        for(int i = 0; i < target.size(); i++){
            std::vector<Point2f> temp_target = rhom_point_cal(target[i]);
            target_armour_rioh.push_back(temp_target);
            draw_rect(temp_target, img, Scalar(255,255,0));
            circle(img, temp_target[4], 5, Scalar(255,255,0), -1);
        }
        
        imshow("result", img);
    }

    waitKey(0);  

 
    return 0;
}