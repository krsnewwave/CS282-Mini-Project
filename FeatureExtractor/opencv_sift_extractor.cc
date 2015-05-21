/* 
 * File:   OpenCVSIFTDescExtractor.cpp
 * Author: dylan
 * 
 * Created on May 13, 2015, 2:53 PM
 */

#include <opencv2/features2d/features2d.hpp>

#include "opencv_sift_extractor.h"

using namespace cv;
using namespace std;

OpenCVSIFTDescExtractor::OpenCVSIFTDescExtractor(int nfeatures,
        int nOctaveLayers, double contrastThreshold, double edgeThreshold,
        double sigma) {
//    detector = SiftDescriptorExtractor(nfeatures, nOctaveLayers,
//            contrastThreshold);
    SiftDescriptorExtractor detector;
}

OpenCVSIFTDescExtractor::~OpenCVSIFTDescExtractor() {
}

/**
 * Return 
 * @param image
 * @return 
 */
void OpenCVSIFTDescExtractor::getSIFTDescriptor(const Mat& image, Mat& dst,
        vector<KeyPoint>& keypoints) {
    detector.detect(image, keypoints);
    detector.compute(image, keypoints, dst);
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

void OpenCVSIFTDescExtractor::process_images(const vector<Mat>& images,
        const cv::Mat& dict, Mat& dst) {
    for (int i = 0; i < images.size(); i++) {
        Mat sift_descriptors;
        vector<KeyPoint> keyPoints;
        getSIFTDescriptor(images[i], sift_descriptors, keyPoints);

        //create a nearest neighbor matcher
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match(sift_descriptors, dict, matches);

        //group to bins
        float bins[dict.rows];

        //update number of bins
        for (int i = 0; i < matches.size(); i++) {
            bins[matches.at(i).trainIdx] = bins[matches.at(i).trainIdx] + 1;
        }

        Mat norm_bins(1, dict.rows, CV_32F, &bins);
        normalize(norm_bins, norm_bins, 0, 1, NORM_MINMAX, -1, Mat());
        dst.push_back(norm_bins);
    }
}