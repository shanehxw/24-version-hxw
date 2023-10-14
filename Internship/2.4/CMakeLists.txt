cmake_minimum_required(VERSION 3.1.0)

project(armour_project)
 
find_package(OpenCV REQUIRED)

add_executable(ar armour_rec.cpp)

add_subdirectory(Mylib1)

target_link_libraries(
    ar                              
    PUBLIC ${OpenCV_LIBS} 
    PUBLIC Mylib1
)

target_include_directories(
    ar                            
    PUBLIC ${Opencv_INCLUDE_DIRS}     
)