aux_source_directory(src M2_DIR)

add_library(M2 SHARED ${M2_DIR})

#find_package(OpenCV REQUIRED)

target_include_directories(
    M2 
    PUBLIC include
)

target_link_libraries(
    M2
    PUBLIC kalman  # try
    PUBLIC A1
    PUBLIC A2
)