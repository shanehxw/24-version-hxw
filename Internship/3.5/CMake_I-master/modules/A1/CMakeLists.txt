aux_source_directory(src A1_DIR)

add_library(A1 SHARED ${A1_DIR})

#find_package(OpenCV REQUIRED)

target_include_directories(
    A1 
    PUBLIC include
)
