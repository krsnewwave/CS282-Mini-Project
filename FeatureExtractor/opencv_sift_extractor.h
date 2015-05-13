/* 
 * File:   OpenCVSIFTDescExtractor.h
 * Author: dylan
 *
 * Created on May 13, 2015, 2:53 PM
 */

#ifndef OPENCVSIFTDESCEXTRACTOR_H
#define	OPENCVSIFTDESCEXTRACTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

class OpenCVSIFTDescExtractor {
public:
    OpenCVSIFTDescExtractor(int nfeatures = 0,
        int nOctaveLayers = 3, double contrastThreshold = 0.04,
        double edgeThreshold = 10, double sigma = 1.6);
    virtual ~OpenCVSIFTDescExtractor();
    cv::Mat getSIFTDescriptor(cv::Mat image);
    cv::Mat equalizeUsingYCBCR(cv::Mat img);
private:
    cv::SiftDescriptorExtractor detector;
};
#endif	/* OPENCVSIFTDESCEXTRACTOR_H */