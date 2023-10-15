cmake_minimum_required(VERSION 3.1.0)

project(Power_machine_project)
 
find_package(OpenCV REQUIRED)

add_executable(pm pm_detect.cpp)

add_subdirectory(Mylib1_rec)

target_link_libraries(
    pm                              
    PUBLIC ${OpenCV_LIBS}
    PUBLIC Mylib1_rec
)

target_include_directories(
    pm                            
    PUBLIC ${Opencv_INCLUDE_DIRS}     
)