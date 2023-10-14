aux_source_directory(src M1_DIR)

add_library(M1 SHARED ${M1_DIR})

#find_package(OpenCV REQUIRED)

target_include_directories(
    M1 
    PUBLIC include
)

target_link_libraries(
    M1                              
    PUBLIC A1
)