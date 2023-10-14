add_library(singleton INTERFACE)

target_include_directories(
    singleton
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)
#find_package(OpenCV REQUIRED)

#target_link_libraries(
#    singleton
#    INTERFACE 
#)