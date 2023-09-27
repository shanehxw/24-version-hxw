----------------------------------------CMakeLists简单使用-----------------------------------------

cmake_minimum_required(VERSION 3.1.0)  # 版本管理

project(test_project)  # project命名
 
find_package(OpenCV REQUIRED)  # 解析Findxxx.cmake / xxxConfig.cmake / xxx-Config.cmake
                               # 对于Findxxx，查找CMAKE_MODULE_PATH then CMAKE_ROOT

add_executable(demo test.cpp)  # 生成可执行文件

target_link_libraries(
    demo                              # target
    PUBLIC ${OpenCV_LIBS}             # link(PUBLIC or PRIVATE or INTERFACE)
    ...
)

target_include_directories(
    demo                              # target
    PUBLIC ${Opencv_INCLUDE_DIRS}     # head(PUBLIC or PRIVATE or INTERFACE)
)


aux_source_directory(MY_LIB_DIRS src) :创建cpp文件集合 cpp文件位置

add_library(Mylib1 SHARED MY_LIB_DIRS) : 库 动态链接/静态链接 cpp文件集合

add_subdirectory(...) : 添加子模块

#-------------------------------------------命令行--------------------------------------------------

# cd build  # 准备在build文件夹里面放置文件
# cmake ..  # 生成makefile文件  
# make / cmake --build .  # 生成可执行文件
# rm -rf build # 删除build
# cmake-gui ..  # 打开gui界面






-------------------------------------------基本函数-----------------------------------------------

全局变量（缓存变量）：CACHE + 变量类型（如INTERNAL、BOOL、PATH）
引用变量：${AAA} == "abc"
动态库 xxx.so  静态库 xxx.a

set(AAA "abc" CACHE INTENRAL ""):变量名 变量内容 格式 注释  # 可以有多个值 
unset（AAA CHAHE）  # 只有缓存变量写CACHE

message（） # 