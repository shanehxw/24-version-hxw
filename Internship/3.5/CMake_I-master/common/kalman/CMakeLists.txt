set(kalman_INCLUDE_DIR)
list(APPEND kalman_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include")
list(APPEND kalman_INCLUDE_DIR "${OpenCV_INCLUDE_DIRS}")

set(kalman_INCLUDE_DIRS ${kalman_INCLUDE_DIR}
                        CACHE PATH "kalman filter include directories")


add_library(kalman INTERFACE)

target_include_directories(
    kalman 
    INTERFACE include
)

find_package(OpenCV REQUIRED)

target_link_libraries(
    kalman
    INTERFACE ${OpenCV_LIBS} 
)