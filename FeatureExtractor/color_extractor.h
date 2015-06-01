/* 
 * File:   ColorFeatureExtractor.h
 * Author: dylan
 *
 * Created on May 24, 2015, 5:31 PM
 */

#ifndef COLORFEATUREEXTRACTOR_H
#define	COLORFEATUREEXTRACTOR_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv_lbp_extractor.h"
#include "opencv_sift_extractor.h"

class ColorFeatureExtractor {
public:
    ColorFeatureExtractor();
    virtual ~ColorFeatureExtractor();
    void getPatchColorDescriptors(const cv::Mat& img,
            const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
            int patchSize = 15);
    void getColorDescriptors(cv::Mat& img, cv::Mat& descriptors);

private:
    cv::Mat equalizeIntensity(const cv::Mat& inputImage);
};

#endif	/* COLORFEATUREEXTRACTOR_H */

