/* 
 * File:   opencv_lbp_extractor.h
 * Author: dylan
 *
 * Created on May 14, 2015, 10:43 PM
 */

#ifndef OPENCV_LBP_EXTRACTOR_H
#define	OPENCV_LBP_EXTRACTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

class OpenCVLBPDescExtractor {
public:
    OpenCVLBPDescExtractor();
    virtual ~OpenCVLBPDescExtractor();
    cv::Mat getOLBPDescriptor(const cv::Mat& image, cv::Mat& dst,
            const cv::vector<cv::KeyPoint>& keyPoints, int patchSize = 8);
private:
    template <typename _Tp> void lbp(const cv::Mat& src, cv::Mat& dest);
};

#endif	/* OPENCV_LBP_EXTRACTOR_H */

