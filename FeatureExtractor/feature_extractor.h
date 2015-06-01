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
#include "opencv2/contrib/contrib.hpp"
#include <opencv2/ml/ml.hpp>

#include "sampler.h"

using namespace std;

class FeatureExtractor {
public:
    FeatureExtractor();
    virtual ~FeatureExtractor();
    cv::Mat extractSIFTFeatures(string sample_file, string imgdir);
    cv::Mat extractSIFTFeatures(const std::vector<cv::Mat>& images);
    //    cv::Mat extractLBPFeatures(string sample_file, string imgdir);
    //    cv::Mat extractLBPFeatures(const std::vector<cv::Mat>& images);
    cv::Mat extractSIFTLBPFeatures(const std::vector<cv::Mat>& images);

    cv::Mat create_dictionary(cv::Mat features);
    cv::Mat createSiftTrainingDescriptors(Dataset ds, cv::Mat dict,
            const std::map<std::string, int>& categoryMap, vector<int>& training_classes);
    cv::Mat createSiftLBPTrainingDescriptors(Dataset ds, cv::Mat dict,
            const std::map<std::string, int>& categoryMap, vector<int>& training_classes);
    cv::Ptr<cv::FaceRecognizer> trainLBPModel(string sample_file, string imgdir,
            const map<string, int>& categoryMap, vector<int>& classes);
    cv::Size getSize();
    void setSize(cv::Size size);
    
private:
    cv::Size resizeSize = cv::Size();
};


#endif	/* FEATURE_EXTRACTOR_H */

