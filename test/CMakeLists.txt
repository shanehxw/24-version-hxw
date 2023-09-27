cmake_minimum_required(VERSION 3.1.0)

project(test_project)
 
find_package(OpenCV REQUIRED)

add_executable(demo test.cpp)

target_link_libraries(
    demo                              
    PUBLIC ${OpenCV_LIBS} 
)

