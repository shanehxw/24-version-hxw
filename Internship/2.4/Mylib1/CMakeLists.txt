aux_source_directory(src MY_LIB_DIR1)

add_library(Mylib1 SHARED ${MY_LIB_DIR1})

#find_package(OpenCV REQUIRED)

target_include_directories(
    Mylib1 PUBLIC include

)

# 链接第三方库

target_link_libraries(
    Mylib1 PUBLIC ${OpenCV_LIBS} 

)
