cmake_minimum_required(VERSION 3.1.0)

project(Power_machine_project)
 
find_package(OpenCV REQUIRED)

add_executable(pm pm_detect.cpp)

target_link_libraries(
    pm                              
    PUBLIC ${OpenCV_LIBS} 
)

target_include_directories(
    pm                            
    PUBLIC ${Opencv_INCLUDE_DIRS}     
)