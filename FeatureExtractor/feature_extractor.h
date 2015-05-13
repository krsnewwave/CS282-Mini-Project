/* 
 * File:   feature_extractor.h
 * Author: dylan
 *
 * Created on May 13, 2015, 1:34 PM
 */

#ifndef FEATURE_EXTRACTOR_H
#define	FEATURE_EXTRACTOR_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <features.h>
#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <stdio.h>

#include "opencv_sift_extractor.h"
#include "sampler.h"

using namespace std;

class FeatureExtractor {
public:
    FeatureExtractor();
    virtual ~FeatureExtractor();
    cv::Mat extract_features(string sample_file, string imgdir);
    cv::Mat create_dictionary(cv::Mat features);
    cv::Mat create_training_descriptors(Dataset ds, cv::Mat dict);
};


#endif	/* FEATURE_EXTRACTOR_H */

