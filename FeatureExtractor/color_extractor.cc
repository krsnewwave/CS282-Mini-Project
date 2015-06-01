/* 
 * File:   ColorFeatureExtractor.cc
 * Author: dylan
 * 
 * Created on May 24, 2015, 5:31 PM
 */

#include <opencv2/core/types_c.h>

#include "color_extractor.h"
#include "utils.cc"

using namespace std;
using namespace cv;

ColorFeatureExtractor::ColorFeatureExtractor() {
}

ColorFeatureExtractor::~ColorFeatureExtractor() {
}

Mat ColorFeatureExtractor::equalizeIntensity(const Mat& inputImage) {
    if (inputImage.channels() >= 3) {
        Mat ycrcb;
        cvtColor(inputImage, ycrcb, CV_BGR2YCrCb);

        vector<Mat> channels;
        split(ycrcb, channels);

        equalizeHist(channels[0], channels[0]);

        Mat result;
        merge(channels, ycrcb);
        cvtColor(ycrcb, result, CV_YCrCb2BGR);

        return result;
    }
    return Mat();
}

void ColorFeatureExtractor::getPatchColorDescriptors(const Mat& img,
        const vector<KeyPoint>& keypoints, Mat& descriptors, int patchSize) {
    Mat patch;
    Size size = Size(patchSize, patchSize); //square patchSize x patchSize region (8x8 patch)
    Mat imgEq = equalizeIntensity(patch);
    for (unsigned int i = 0; i < keypoints.size(); i++) {
        Point2f center(keypoints.at(i).pt.x, keypoints.at(i).pt.y);
        getRectSubPix(img, size, center, patch);
        getColorDescriptors(patch, descriptors);
    }
}

void ColorFeatureExtractor::getColorDescriptors(Mat& img, Mat& descriptors) {
    Mat out;
    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    int histSize = 256; //from 0 to 255

    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256}; //the upper boundary is exclusive
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    hconcat(b_hist.t(), g_hist.t(), out);
    hconcat(out, r_hist.t(), out);

    descriptors.push_back(out);
}


