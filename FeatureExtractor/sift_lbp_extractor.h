/* 
 * File:   OpenCVSIFTLBPExtractor.h
 * Author: dylan
 *
 * Created on May 24, 2015, 1:55 PM
 */

#ifndef OPENCVSIFTLBPEXTRACTOR_H
#define	OPENCVSIFTLBPEXTRACTOR_H

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
#include "color_extractor.h"

class OpenCVSIFTLBPExtractor {
public:
    OpenCVSIFTLBPExtractor();
    virtual ~OpenCVSIFTLBPExtractor();
    void create_sift_lpb_dictionary(const std::vector<cv::Mat>& images,
            OpenCVSIFTDescExtractor siftExtractor,
            OpenCVLBPDescExtractor lbpExtractor,
            cv::Mat& dst);
    void process_sift_lbp_images(const std::vector<cv::Mat>& images,
            const cv::Mat& dict, cv::Mat& dst);
    void create_sift_lpb_color_dictionary(const std::vector<cv::Mat>& images,
            OpenCVSIFTDescExtractor siftExtractor,
            OpenCVLBPDescExtractor lbpExtractor,
            ColorFeatureExtractor colorExtractor,
            cv::Mat& dst);
    void process_sift_lbp_color_images(const std::vector<cv::Mat>& images,
            const cv::Mat& dict, cv::Mat& dst);
private:

};

#endif	/* OPENCVSIFTLBPEXTRACTOR_H */

