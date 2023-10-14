#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp> 
//#include <opencv2/videoio.hpp> 
//#include <opencv2/highgui.hpp> 

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


//-------------------------------------------------工具库----------------------------------------------------

void find_width_and_height(float width, float height, float* array)  // 对minareaRect的宽高进行排序
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

void draw_rect(std::vector<Point2f> rect, Mat frame)  // 通过vector<Point2f>画出矩形
{   
    for(int i = 0; i < 4; i++)
    {   
        line(frame, rect[i%4], rect[(i+1)%4], Scalar(255,0,120), 1);
    }
}

void draw_point(std::vector<Point2f> points, Mat frame, Scalar color, int r)  // 通过std::vector<Point2f>画出点
{
    for(int i = 0; i < points.size(); i++)
    {
         circle(frame, points[i], r, color, -1);
    }
}

float calculateDistance(cv::Point2f pt1, cv::Point2f pt2)   // 通过两个Point2f计算距离
{  

    float dx = pt2.x - pt1.x;  

    float dy = pt2.y - pt1.y;  

    return std::sqrt(dx * dx + dy * dy);  

} 

//------------------------------------------------排序工具----------------------------------------------------

std::vector<RotatedRect> compare_Left_or_Right(const RotatedRect rect_1, const RotatedRect rect_2)  // 通过中心点对矩形进行从左到右的排序
{   
    std::vector<RotatedRect> left_then_right;
	if (rect_1.center.x >= rect_2.center.x)
    {
        left_then_right.push_back(rect_2);
        left_then_right.push_back(rect_1);
		return left_then_right;
    }
	else
    {
		left_then_right.push_back(rect_1);
        left_then_right.push_back(rect_2);
        return left_then_right;
    }
}

//------------------------------------------------预处理:通道分离+滤波--------------------------------------------------

Mat img_gray(Mat frame)  // 灰度图转化
{   
    Mat dst_gray;
    Mat dst_baw;
    cv::cvtColor(frame, dst_gray, COLOR_BGR2GRAY);
    //imshow("gray",dst_gray);
    cv::threshold(dst_gray, dst_baw, 30, 255, cv::THRESH_BINARY);
    //imshow("gray_baw",dst_baw);

    return dst_baw;

}

Mat img_channel(Mat frame)  // BGR通道处理
{
    std::vector<cv::Mat> bgr_channels;
                
    cv::split(frame, bgr_channels);

    //通道相减
    Mat imageChannelcal = bgr_channels[2] - bgr_channels[0];

    //-------高斯低通滤波------
    //Mat frame_gaublur;
    //cv::GaussianBlur(imageChannelcal, frame_gaublur, Size(3,3), 0, 0);  // 高斯滤波处理
    

    //-------二值化-------
    cv::Mat dst_baw;
    cv::threshold(imageChannelcal, dst_baw, 80, 255, cv::THRESH_BINARY);
    //imshow("baw",dst_baw);

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

//-------滤波处理（矩形膨胀）-------
Mat img_filter_forrect(Mat dst){
    
    //-------高斯低通滤波------
    //Mat frame_gaublur;
    //cv::GaussianBlur(imageChannelcal, frame_gaublur, Size(3,3), 0, 0);  // 高斯滤波处理

    //-------形态学开闭操作-------
    // 使连成整体
    Mat dst_dil;
    Mat dst_ero;
    Mat white_hat;
    cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));  //池化核大小
    cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));  //池化核大小
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    //cv::dilate(dst, dst_dil, kernel7, Point(-1,-1), 1);
    //cv::erode(dst_dil, dst_ero, kernel7, Point(-1,-1), 1);
    
    //cv::dilate(dst, dst_dil, kernel5, Point(-1,-1), 1);
    //cv::erode(dst_dil, dst_ero, kernel5, Point(-1,-1), 1);
    
    cv::dilate(dst, dst_dil, kernel3, Point(-1,-1), 1);
    //cv::erode(dst_dil, dst_ero, kernel3, Point(-1,-1), 1);
    //imshow("dilate and erode", dst_ero);
    
    Mat frame_gaublur;
    Mat frame_blur;
    cv::GaussianBlur(dst_dil, frame_gaublur, Size(5,5), 0, 0);  // 高斯滤波处理
    cv::GaussianBlur(frame_gaublur, frame_gaublur, Size(5,5), 0, 0);  // 高斯滤波处理
    cv::GaussianBlur(frame_gaublur, frame_gaublur, Size(5,5), 0, 0);  // 高斯滤波处理
    cv::GaussianBlur(frame_gaublur, frame_gaublur, Size(5,5), 0, 0);  // 高斯滤波处理
    //cv::GaussianBlur(frame_gaublur, frame_gaublur, Size(5,5), 0, 0);  // 高斯滤波处理
    
    //blur(frame_gaublur, frame_blur, Size (5,5));
    //blur(frame_blur, frame_blur, Size (5,5));



    return frame_gaublur;
}

//-------滤波处理（靶心腐蚀）-------
Mat img_filter_forcircle1(Mat dst){

    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));  //池化核大小
    cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    cv::GaussianBlur(dst, dst, Size(3,3), 0, 0);
    cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    cv::dilate(dst, dst, kernel3, Point(-1,-1), 1);
    cv::erode(dst, dst, kernel3, Point(-1,-1), 1);

    //cv::GaussianBlur(dst, dst, Size(3,3), 0, 0);
    //cv::erode(dst, dst, kernel3, Point(-1,-1), 1);
    
    imshow("erode", dst);
    return dst;
}

//-------在轮廓中搜索，返回只有一级子轮廓以及没有子轮廓的轮廓
std::vector<std::vector<Point>> find_target_contours0_1(std::vector<std::vector<Point>> contours, std::vector<cv::Vec4i> hierarchy)
{
    std::vector<std::vector<Point>> result;

    for (size_t i = 0; i < contours.size(); i++)
    {
        for (int j = hierarchy[i][2]; j != -1; j = hierarchy[j][0])  // j是i的一个子轮廓，遍历i的所有一级子轮廓
        {
            if (hierarchy[j][2] == -1)  // 如果子轮廓没有子轮廓->i只有一级子轮廓
            {
                result.push_back(contours[i]);
            }
            else  //  如果子轮廓j有子轮廓->不管，反正迟早会遍历到
            {   
                continue;
            }
        }

        if (hierarchy[i][2] == -1)  // 如果i没有子轮廓
            result.push_back(contours[i]);
    }

    return result;
}

//-------寻找没有子轮廓的轮廓
std::vector<std::vector<Point>> find_target_contours0(std::vector<std::vector<Point>> contours, std::vector<cv::Vec4i> hierarchy)
{
    std::vector<std::vector<Point>> result;

    for (size_t i = 0; i < contours.size(); i++)
    {
        if (hierarchy[i][2] == -1)  // 如果i没有子轮廓
            result.push_back(contours[i]);
    }

    return result;
}

//---------------------------------------------判断是否为激活靶心----------------------------------------------------------------------------

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
    for(i = 0; i < rect_center.size(); i++){
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

//---------------------------------------------轮廓检测——靶心——腐蚀----------------------------------------------------------

//-------邻近点取方差聚合(初步)-------
Point2f near_fix(std::vector<Point2f> group, int r)
{   
    float sum_x = 0;
    float sum_y = 0;
    float avg_x = 0;
    float avg_y = 0;

    int num = group.size();
    
    float point_x = 0;
    float point_y = 0;
    for(int i = 0; i  < num; i++){
        sum_x += group[i].x;
        sum_y += group[i].y;
    }
    avg_x = sum_x / num;
    avg_y = sum_y / num;

    float k_array[num] = {0};
    float k_sum = 0;
    Point2f pre_center(avg_x, avg_y);
    for(int i = 0; i  < num; i++){
        k_array[i] = 1 - ( calculateDistance(group[i], pre_center) / r);
    }

    for(int i = 0; i  < num; i++){
        point_x += k_array[i] * group[i].x;
        point_y += k_array[i] * group[i].y;
        k_sum += k_array[i];
    }

    Point2f center(point_x / k_sum, point_y / k_sum);

    //std::cout<< center.x << "||" << center.y << std::endl;
    return center;
}

//-------聚合主函数-------
std::vector<Point2f> point_fix(std::vector<Point2f> points, float k1, float k2)  // 与较近点的范围阈值
{   
    //std::cout<<points.size()<<std::endl;
    std::vector<Point2f> center_points;
    int i = 0;
    int j = 0;

    float k3 = k1 * 1.732;
    float k4 = k2 * 1.732;  // 与较远点的范围阈值，*1.732


    float near_dis = 15;  // 与相近点的范围阈值


    int if_group = 0;  // 用来判断是否处是接近点
    int group_num[(int)points.size()] = {0};  // 用来储存相近点在points里的序号，避免for(i)重复进入possible_group

    for(i = 0, if_group = 0; i < points.size(); i++){
        
        if(group_num[i] == 1)  // 判断是否已经为接近点
        {
            continue;
        }

        int normal_num = 0;
        int wrong_num = 0;  // 计数器，用来判断是否是无关点
        float k_judge = 1.5;

        std::vector<Point2f> possible_group;  // 用于存储需要预处理的接近点

        for(j = 0; j < points.size(); j++){
            
            if(i != j){
                float dis = calculateDistance(points[i], points[j]);
                std::cout<<i<<"、dis = "<<dis<<std::endl;
                
                if( (k1 < dis && dis < k2) || (k3 < dis && dis < k4)){  // 在正常范围
                        normal_num++;
                }
                else if(dis < near_dis){  // 若有在极小范围内的
                    if_group++;
                    if(if_group == 1)
                    {
                        possible_group.push_back(points[i]);
                        possible_group.push_back(points[j]);  // 第一次存，存入两个点
                        group_num[i] = 1;
                        group_num[j] = 1;
                    }
                    if(if_group >= 1)
                    {
                        possible_group.push_back(points[j]);  // 第n次存，存入一个点
                        group_num[j] = 1;
                    }
                
                }
                else  // 异常范围
                    wrong_num++;
            }
            
        }
        
        if(normal_num > wrong_num)
        {   
            if(if_group > 0)
            {   
                Point2f fix_point = near_fix(possible_group, near_dis);
                if(fix_point.x > VIDEO_WIDTH/10 && fix_point.x < VIDEO_WIDTH * 9/10 && fix_point.y > VIDEO_HEIGHT/10 && fix_point.y < VIDEO_HEIGHT* 9/10)
                    center_points.push_back( fix_point );
                else
                    continue;
            }
            else
            {   
                if(points[i].x > VIDEO_WIDTH/10 && points[i].x < VIDEO_WIDTH * 9/10 && points[i].y > VIDEO_HEIGHT/10 && points[i].y < VIDEO_HEIGHT* 9/10)
                    center_points.push_back( points[i] );
                else
                    continue;
            }
        }
        else
            continue;
    }
    
    //if(center_points.size() > 5)
        //center_points = point_fix(center_points, k1+3, k2-3);
    return center_points;
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
    cv::Mat empty_Mat = cv::Mat::zeros(frame.size(), frame.type());
    drawContours(empty_Mat, contours, -1, Scalar(255, 0, 255), -1);
    imshow("drawframe",empty_Mat);

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



//---------------------------------------------轮廓检测——矩形对、R标——膨胀-----------------------------------------------------

//-------矩形和R标检测-------（待改进）
// 3、记录R标位置点
std::vector<Point2f> rect_contours_pre_recognize(std::vector<std::vector<cv::Point> > contours, Mat frame, Point2f* center_point)
{
    std::vector<cv::RotatedRect> contours_min_rects;  //预筛选轮廓的最小外接矩形
    std::vector<float*> minrect_width_height;  // 顺便记录已经排序好的长和宽

    float tempminrect_width_height[2] = {0, 0};
    
    for (unsigned int contour_index = 0; contour_index < contours.size(); contour_index++)
    {   
        cv::RotatedRect temp_minrect = minAreaRect(contours[contour_index]);  // 暂存矩形
        tempminrect_width_height[0] = 0;
        tempminrect_width_height[1] = 0;
        find_width_and_height(temp_minrect.size.width, temp_minrect.size.height, tempminrect_width_height);
        // 寻找激活矩形
        if (temp_minrect.size.area() <= 3000.0 && temp_minrect.size.area() > 800)
        {
            if(tempminrect_width_height[0] / tempminrect_width_height[1] < 5 && tempminrect_width_height[0] / tempminrect_width_height[1] > 2)  // 长宽筛选
                {
                contours_min_rects.push_back(temp_minrect);
                minrect_width_height.push_back(tempminrect_width_height);
                }
        }
        //寻找R标
        if(temp_minrect.center.x > VIDEO_WIDTH * 3/7 && temp_minrect.center.x < VIDEO_WIDTH * 4/7 && temp_minrect.center.y > VIDEO_HEIGHT * 3/7 && temp_minrect.center.y < VIDEO_HEIGHT * 4/7)
        {
            if(temp_minrect.size.area() <= 1200.0 && temp_minrect.size.area() > 400)
                {
                    if(tempminrect_width_height[0] / tempminrect_width_height[1] < 1.2)
                    {
                        *center_point = temp_minrect.center;
                        //circle(frame, center_point, 4, Scalar(255,255,255), -1);
                    }
                }
        }
    } 

    //for(int minrect_index = 0; minrect_index < contours_min_rects.size(); minrect_index++)
    //{   
    //    Point2f min_rect_point[4];
    //    contours_min_rects[minrect_index].points(min_rect_point);
    //    draw_rect_array(min_rect_point, frame, Scalar(255, 0, 0));
    //}
    //imshow("min_rect",frame);

    //取出得到的矩形中心点
    std::vector<Point2f> rect_center;
    for(int i = 0; i < contours_min_rects.size(); i++){
        rect_center.push_back(contours_min_rects[i].center);
    }
    return rect_center;
}

//-------进行两两分组-------//目前来看，由于检测矩形的不稳定，两两分组不太能对靶心检测进行辅助，因此暂时不使用
//-------得到的矩形按照顺序成对出现-------
std::vector<std::vector<RotatedRect>> rect_compare(std::vector<cv::RotatedRect> pre_minrect, Mat frame, std::vector<Point2f> judge_point)
{   
    std::vector<std::vector<RotatedRect>> rect_group;
    //直接开始遍历，然后分组
    int rect_index1 = 0;
    for(rect_index1 = 0; rect_index1 < (pre_minrect.size() - 1); ++rect_index1)
        for(int rect_index2 = rect_index1 + 1; rect_index2 < pre_minrect.size(); ++rect_index2)
        {   
            float dis = calculateDistance(pre_minrect[rect_index1].center, pre_minrect[rect_index2].center);
            if(dis < 80)
            {   
                Point2f mid_point ( (pre_minrect[rect_index1].center.x + pre_minrect[rect_index2].center.x)/2, (pre_minrect[rect_index1].center.y + pre_minrect[rect_index2].center.y)/2 );
                std::vector<RotatedRect> temp_group;
                //前左后右进入容器
                rect_group.push_back( compare_Left_or_Right(pre_minrect[rect_index1], pre_minrect[rect_index2]) );  //
                
            }
        
        }
    for(int group_index = 0; group_index < rect_group.size(); group_index++)
    {   
        std::vector<RotatedRect> draw_group = rect_group[0];
        Point2f temp_point0 [4];
        Point2f temp_point1 [4];
        draw_group[0].points(temp_point0);
        draw_group[1].points(temp_point1);
        
        draw_rect_array(temp_point0, frame, Scalar(255,255,0));
        draw_rect_array(temp_point1, frame, Scalar(255,255,0));
        circle(frame, draw_group[0].center, 3, Scalar(255,255,255), -1);
        circle(frame, draw_group[1].center, 3, Scalar(255,255,255), -1);
    }
    imshow("circle",frame);
    return rect_group;
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

//------------------------------------------------主函数--------------------------------------------------
int main(){
    std::cout<<"Hello,pm"<<std::endl;

    cv::VideoCapture cap("../Power_machine.avi");

    Mat frame;

    if(cap.isOpened())
    {

        while(cap.read(frame))
        {   
            //imshow("ori",frame);

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
            imshow("target", frame);
            


            if(cv::waitKey(50) >= 0)
                break;
        }


    }

    return 0;
}