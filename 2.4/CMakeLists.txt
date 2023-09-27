cmake_minimum_required(VERSION 3.1.0)

project(armour_project)
 
find_package(OpenCV REQUIRED)

add_executable(ar armour_rec.cpp)

target_link_libraries(
    ar                              
    PUBLIC ${OpenCV_LIBS} 
)

target_include_directories(
    ar                            
    PUBLIC ${Opencv_INCLUDE_DIRS}     
)