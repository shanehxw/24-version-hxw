aux_source_directory(src MODULE2_DIR)

add_library(module2 SHARED ${MODULE2_DIR})

find_package(OpenCV REQUIRED)

target_include_directories(
    module2 
    PUBLIC include
    PUBLIC include/robotlab
    PUBLIC include/robotlab/opcua_cs
    PUBLIC include/robotlab/opcua_cs/ua_type
)

# 链接第三方库

target_link_libraries(
    module2
    PUBLIC ${OpenCV_LIBS}
    PUBLIC pthread 
    
)