cmake_minimum_required(VERSION 3.1.0)

project(armour_pro_project)
 
find_package(OpenCV REQUIRED)

add_executable(arp armour_pro.cpp)

target_link_libraries(
    arp                              
    PUBLIC ${OpenCV_LIBS} 
)

target_include_directories(
    arp                            
    PUBLIC ${Opencv_INCLUDE_DIRS}     
)