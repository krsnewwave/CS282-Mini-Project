/* 
 * File:   opencv_lbp_extractor.cc
 * Author: dylan
 * 
 * Created on May 14, 2015, 10:43 PM
 */

#include "opencv_lbp_extractor.h"

using namespace cv;

OpenCVLBPDescExtractor::OpenCVLBPDescExtractor() {
}

OpenCVLBPDescExtractor::~OpenCVLBPDescExtractor() {
}

template <typename _Tp> void OpenCVLBPDescExtractor::lbp(
        const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            _Tp center = src.at<_Tp>(i, j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i - 1, j - 1) > center) << 7;
            code |= (src.at<_Tp>(i - 1, j) > center) << 6;
            code |= (src.at<_Tp>(i - 1, j + 1) > center) << 5;
            code |= (src.at<_Tp>(i, j + 1) > center) << 4;
            code |= (src.at<_Tp>(i + 1, j + 1) > center) << 3;
            code |= (src.at<_Tp>(i + 1, j) > center) << 2;
            code |= (src.at<_Tp>(i + 1, j - 1) > center) << 1;
            code |= (src.at<_Tp>(i, j - 1) > center) << 0;
            dst.at<unsigned char>(i - 1, j - 1) = code;
        }
    }
}

Mat OpenCVLBPDescExtractor::getOLBPDescriptor(const Mat& image, Mat& dst,
        const std::vector<KeyPoint>& keyPoints,int patchSize) {
    Mat lbpimg, patch, lbp_descriptors;
    lbp<unsigned char>(image, dst);
    Size size = Size(patchSize, patchSize);
     for(int i=0; i<keyPoints.size();i++){
        Point2f center(keyPoints.at(i).pt.x,keyPoints.at(i).pt.y);
        getRectSubPix(lbpimg,size,center,patch);
        patch = patch.reshape(0,1); //flatten matrix to 1-D Vector
        lbp_descriptors.push_back(patch);
    }
    lbp_descriptors.convertTo(dst,CV_32FC1);
}