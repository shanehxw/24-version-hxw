aux_source_directory(src MY_LIB_REC_DIR1)

add_library(Mylib1_rec SHARED ${MY_LIB_REC_DIR1})

#find_package(OpenCV REQUIRED)

target_include_directories(
    Mylib1_rec PUBLIC include

)

# 链接第三方库

target_link_libraries(
    Mylib1_rec PUBLIC ${OpenCV_LIBS} 

)
