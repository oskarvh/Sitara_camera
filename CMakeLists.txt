cmake_minimum_required(VERSION 2.8)

project(camera_cv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(camera_cv main.cpp)

target_link_libraries(camera_cv ${OpenCV_LIBS})
