aux_source_directory(src/module1a MODULE1_DIR)
aux_source_directory(src/module1b MODULE1_DIR)

add_library(module1 SHARED ${MODULE1_DIR})

#find_package(OpenCV REQUIRED)

target_include_directories(
    module1 
    PUBLIC include
    PUBLIC include/robotlab/module1

)

# 链接第三方库

target_link_libraries(
    module1 
    PUBLIC assembly2
    
)