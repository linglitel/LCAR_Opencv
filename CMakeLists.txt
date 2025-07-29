cmake_minimum_required(VERSION 4.0)
project(LCAR_Opencv)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

set(RKNN_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/rknn)
include_directories(${RKNN_SDK_PATH}/include)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(LCAR_Opencv
        src/main.cpp
        src/postprocess.cpp
        src/yolo11.cpp
        include/yolo11.h
)  #[Clion need them]

target_include_directories(LCAR_Opencv
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(LCAR_Opencv
        ${OpenCV_LIBRARIES})

target_link_libraries(LCAR_Opencv
        ${RKNN_SDK_PATH}/librknnrt.so
)