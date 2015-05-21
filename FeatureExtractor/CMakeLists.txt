cmake_minimum_required(VERSION 2.8.4)
project(FeatureExtractor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED core imgproc highgui calib3d 
        features2d nonfree contrib ml)
find_package(OpenGL REQUIRED)

#add_library(utils utils.h utils.cc)

set(EXTRACTOR_FILES feature_extractor.cc feature_extractor.h sampler.cc sampler.h
        opencv_sift_extractor.cc opencv_sift_extractor.h 
        opencv_lbp_extractor.h opencv_lbp_extractor.cc)
add_executable(feature_extractor ${EXTRACTOR_FILES})
target_link_libraries(feature_extractor ${OpenCV_LIBS})

set(EVALUATOR_FILES evaluator.cc evaluator.h sampler.cc sampler.h
opencv_sift_extractor.cc opencv_sift_extractor.h
opencv_lbp_extractor.h opencv_lbp_extractor.cc)
add_executable(evaluator ${EVALUATOR_FILES})
target_link_libraries(evaluator ${OpenCV_LIBS})

