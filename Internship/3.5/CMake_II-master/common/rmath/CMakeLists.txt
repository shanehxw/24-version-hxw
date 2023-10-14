aux_source_directory(src RMATH_DIR)

add_library(rmath SHARED ${RMATH_DIR})

#find_package(OpenCV REQUIRED)

target_include_directories(
    rmath 
    PUBLIC include
)

# 链接第三方库

target_link_libraries(
    rmath 
    PUBLIC ${OpenCV_LIBS} 

)