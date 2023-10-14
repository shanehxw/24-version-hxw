aux_source_directory(src math_DIR)

add_library(math SHARED ${math_DIR})

find_package(OpenCV REQUIRED)

target_include_directories(
    math
    PUBLIC include
)

target_link_libraries(
    math                              
    PUBLIC ${OpenCV_LIBS} 
)