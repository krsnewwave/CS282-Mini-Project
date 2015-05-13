cmake_minimum_required(VERSION 2.8.4)
project(FeatureExtractor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED core imgproc highgui calib3d features2d nonfree)
find_package(OpenGL REQUIRED)

#add_library(utils utils.hpp utils.cpp)

set(SOURCE_FILES feature_extractor.cc)
add_executable(feature_extractor ${SOURCE_FILES})
target_link_libraries(feature_extractor ${OpenCV_LIBS})
