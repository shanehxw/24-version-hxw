aux_source_directory(src A2_DIR)

add_library(A2 SHARED ${A2_DIR})

#find_package(OpenCV REQUIRED)

target_include_directories(
    A2 
    PUBLIC include
)