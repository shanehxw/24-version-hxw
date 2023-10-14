cmake_minimum_required(VERSION 3.1.0)

project(Test)
 
find_package(OpenCV REQUIRED)

add_executable(server server.cpp)
add_executable(client client.cpp)

add_executable(assembly1_test ${CMAKE_CURRENT_SOURCE_DIR}/modules/assembly1/test/assembly1_test.cpp)


# # # # 以下内容禁止修改 # # # #
option(BUILD_A "build modules a" ON)
option(BUILD_B "build modules b" OFF)
option(BUILD_TESTS "build tests" OFF)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(modules)
add_subdirectory(common)
# 小提示: module2 需要用到 pthread，即 target_link_libraries(xxx pthread)
# # # # 以上内容禁止修改 # # # #

add_test(NAME assembly1_test COMMAND assembly1_test)

option(with_a ON)
add_definitions(-DWITH_A=${with_a})

target_include_directories(
    server 
    PUBLIC common/singleton/include
)

target_link_libraries(
    server                              
    PUBLIC ${OpenCV_LIBS} 
    PUBLIC rmath
    PUBLIC module1
    PUBLIC module2
    PUBLIC singleton
)

target_link_libraries(
    client                              
    PUBLIC module2
)

target_link_libraries(  
    assembly1_test  
    PUBLIC assembly1
)
