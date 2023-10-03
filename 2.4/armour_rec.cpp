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
    //cv::imshow("filtering", frame_erode);

    return frame_erode;
}

//-------------------------------------轮廓、中心点检测（主要模块1)----------------------------------------------------

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
		if(minAreaRects[i_temp].size.width * minAreaRects[i_temp].size.height >= 30)  // 过滤面积较小的轮廓
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
                double center_point[2] = {0,0};  // 用于装甲板中心点的储存
                double rect_centerpoint [4] = {0,0,0,0};  // 用于矩形中心点的储存，x-y-x-y
                int No_rect[4] = {0,0,0,0};  // 用于记录direct函数中的矩形序号
                Point2f rect_point1[4];
                Point2f rect_point2[4];  // 用于记录矩形的四个点
                std::vector<RotatedRect> minAreaRects(Point_fix.size());  // 用于矩形信息储存

                minAreaRects = direct_centerpoint(dst, center_point, rect_centerpoint, No_rect, Point_fix);
                
                //draw_point(frame, center_point, rect_centerpoint);


        //-------------------------------------测距模块--------------------------------------------
                //-------测距预处理-------
                if(rect_centerpoint[0] > center_point[0])  // 让右边的矩形作为第一个矩形
                {
                    minAreaRects[No_rect[0]].points(rect_point1);
                    minAreaRects[No_rect[1]].points(rect_point2);
                }
                else
                {
                    minAreaRects[No_rect[1]].points(rect_point1);
                    minAreaRects[No_rect[0]].points(rect_point2);
                }
                
                float rect1_width = minAreaRects[No_rect[0]].size.width;
                float rect1_height = minAreaRects[No_rect[0]].size.height;
                
                float rect2_width = minAreaRects[No_rect[1]].size.width;
                float rect2_height = minAreaRects[No_rect[1]].size.height;
                
                std::vector<Point2f> img_input;
                std::vector<Point3f> world_input;

                //for(int i = 0; i < 4; i++){
                  //  circle(frame, rect_point1[i], 5, Scalar(255,0,255),-1);
                    //circle(frame, rect_point2[i], 5, Scalar(255,0,255),-1);
                //}
                //imshow("test",frame);

                //circle(frame, minAreaRects[No_rect[0]].center, 5, Scalar(255,0,255),-1);
                //imshow("test",frame);

                
                if(rect1_width >= rect1_height)  // 右板rect1右偏
                    {
                        img_input.push_back(rect_point1[0]);  // 1-1 右下
                        img_input.push_back(rect_point1[3]);  // 1-4 右上
                    }
                if(rect1_width < rect1_height)  // 右板rect1左偏
                    {
                        img_input.push_back(rect_point1[3]);  // 1-4 右下
                        img_input.push_back(rect_point1[2]);  // 1-3 右上
                    }

                if(rect2_width >= rect2_height)  // 左板rect2右偏
                    {
                        img_input.push_back(rect_point2[1]);  // 2-2 左下
                        img_input.push_back(rect_point2[2]);  // 2-3 左上
                    }
                if(rect2_width < rect2_height)  // 左板rect2左偏
                    {
                        img_input.push_back(rect_point2[0]);  // 2-1 左下
                        img_input.push_back(rect_point2[1]);  // 2-2 左上
                    }
                img_input.push_back(Point2f(center_point[0],center_point[1]));  // 中心

                }

                if(rect_point1[0].x <= center_point[0])  // 装甲板左偏，左边的是rect1
                {   
                    if(rect1_width >= rect1_height)  // 左板rect1右偏
                    {
                        img_input.push_back(rect_point1[1]);  // 1-2 左下
                        img_input.push_back(rect_point1[2]);  // 1-3 左上
                    }
                    if(rect1_width < rect1_height)  // 左板rect1左偏
                    {
                        img_input.push_back(rect_point1[0]);  // 1-1 左下
                        img_input.push_back(rect_point1[1]);  // 1-2 左上
                    }

                    if(rect2_width >= rect2_height)  // 右板rect2右偏
                    {
                        img_input.push_back(rect_point2[0]);  // 1-1 右下
                        img_input.push_back(rect_point2[3]);  // 1-4 右上
                    }
                    if(rect2_width < rect2_height)  // 右板rect2左偏
                    {
                        img_input.push_back(rect_point2[3]);  // 1-4 右下
                        img_input.push_back(rect_point2[2]);  // 1-3 右上
                    }
                    
                    img_input.push_back(Point2f(center_point[0],center_point[1]));  // 中心

                }
                
                for(int i = 0; i<img_input.size();i++){
                   circle(frame,img_input.at(i),5,Scalar(255,0,255),-1);
                }
                imshow("img",frame);

                world_input.push_back(Point3f(ARMOUR_LENGTH/2, -ARMOUR_WIDTH/2, 0));
                world_input.push_back(Point3f(ARMOUR_LENGTH/2, ARMOUR_WIDTH/2, 0));
                world_input.push_back(Point3f(-ARMOUR_LENGTH/2, -ARMOUR_WIDTH/2, 0));
                world_input.push_back(Point3f(-ARMOUR_LENGTH/2, ARMOUR_WIDTH/2, 0));
                world_input.push_back(Point3f(0,0,0));  // 中心
                

                Mat rvec, tvec;
                cv::solvePnPRansac(world_input, img_input, CAMERA_MADRIX, DIST_COEFFS, rvec, tvec);

                Mat Rvec, Tvec;
                rvec.convertTo(Rvec, CV_32F);  // 旋转向量转换格式
	            tvec.convertTo(Tvec, CV_32F); // 平移向量转换格式 

	            Mat_<float> rotMat(3, 3);
	            Rodrigues(Rvec, rotMat);
	            // 旋转向量转成旋转矩阵

                //std::cout<<Tvec<<std::endl;

                Mat distance;
	            distance = -rotMat.inv() * Tvec;
                //for(int i = 0; i<4 ;++i)
                  //  std::cout<<No_rect[i]<<std::endl;  // --不是轮廓的问题
                //std::cout<<center_point[0]<<std::endl;
                //std::cout<<center_point[1]<<std::endl;  // --不是中心点问题
                std::cout<<(int)distance.at<uchar>(1,3)<<std::endl;

                //-------记录第一帧信息，假设其在深度方向上水平-------
                 // 用于记录第一帧（假设深度为零）的中心点image距离
                    
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


                


            }
        if(cv::waitKey(50) >= 0)  //一秒约24帧,按下键盘任意键退出
            break;
        }
    }
    return 0;
}