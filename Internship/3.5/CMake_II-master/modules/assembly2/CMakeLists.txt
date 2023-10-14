aux_source_directory(src ASSEMBLY2_DIR)

add_library(assembly2 SHARED ${ASSEMBLY2_DIR})

find_package(OpenCV REQUIRED)

target_include_directories(
    assembly2 
    PUBLIC include

)

# 链接第三方库

target_link_libraries(
    assembly2 
    PUBLIC ${OpenCV_LIBS} 

)