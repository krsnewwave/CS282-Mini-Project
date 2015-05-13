/* 
 * File:   OpenCVSIFTDescExtractor.cpp
 * Author: dylan
 * 
 * Created on May 13, 2015, 2:53 PM
 */

#include <opencv2/features2d/features2d.hpp>

#include "opencv_sift_extractor.h"

using namespace cv;

OpenCVSIFTDescExtractor::OpenCVSIFTDescExtractor(int nfeatures,
        int nOctaveLayers, double contrastThreshold, double edgeThreshold,
        double sigma) {
    detector = SiftDescriptorExtractor(nfeatures, nOctaveLayers,
            contrastThreshold);
}

OpenCVSIFTDescExtractor::~OpenCVSIFTDescExtractor() {
}

/**
 * Return 
 * @param image
 * @return 
 */
Mat OpenCVSIFTDescExtractor::getSIFTDescriptor(Mat image) {
    Mat sift_descriptors;
    vector<KeyPoint> keypoints;

    detector.detect(image, keypoints);
    detector.compute(image, keypoints, sift_descriptors);
    return sift_descriptors;
}

Mat OpenCVSIFTDescExtractor::equalizeUsingYCBCR(cv::Mat img) {
    Mat ycrcbImg;
    cvtColor(img, ycrcbImg, CV_RGB2YCrCb);
    vector<Mat> channels;
    split(ycrcbImg, channels);

    equalizeHist(channels[0], channels[0]);

    Mat result;
    merge(channels, ycrcbImg);

    cvtColor(ycrcbImg, result, CV_YCrCb2BGR);
    return ycrcbImg;
}