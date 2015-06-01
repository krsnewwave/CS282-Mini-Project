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
#include "opencv2/contrib/contrib.hpp"

class OpenCVLBPDescExtractor {
public:
    OpenCVLBPDescExtractor();
    virtual ~OpenCVLBPDescExtractor();
    void getOLBPDescriptor(const cv::Mat& image, cv::Mat& dst,
            const std::vector<cv::KeyPoint>& keyPoints, int patchSize = 8);
    void getOLBPDescriptorsFromImages(const std::vector<cv::Mat>& images,
            const std::vector<cv::KeyPoint>& keyPoints,
            cv::Mat& dst, int patchSize = 8);
    cv::Ptr<cv::FaceRecognizer> trainLBP(const std::vector<cv::Mat>& images,
            const std::vector<int>& labels, int radius = 1, int neighbors = 8,
            int sgrid_x = 8, int grid_y = 8, double threshold = DBL_MAX);
    void getHistograms(cv::Mat& dst);
private:
    template <typename _Tp> void lbp(const cv::Mat& src, cv::Mat& dest);
    cv::Ptr<cv::FaceRecognizer> lbp_recog;
};

#endif	/* OPENCV_LBP_EXTRACTOR_H */

