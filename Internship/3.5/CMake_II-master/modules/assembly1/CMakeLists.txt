aux_source_directory(src ASSEMBLY1_DIR1)
aux_source_directory(test ASSEMBLY1_DIR1)

add_library(assembly1 SHARED ${ASSEMBLY1_DIR1})

#find_package(OpenCV REQUIRED)

target_include_directories(
    assembly1 
    PUBLIC include/robotlab

)

# 链接第三方库
