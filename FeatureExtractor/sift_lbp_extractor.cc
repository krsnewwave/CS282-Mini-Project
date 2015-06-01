/* 
 * File:   OpenCVSIFTLBPExtractor.cc
 * Author: dylan
 * 
 * Created on May 24, 2015, 1:55 PM
 */

#include "sift_lbp_extractor.h"
using namespace std;
using namespace cv;

OpenCVSIFTLBPExtractor::OpenCVSIFTLBPExtractor() {
}

OpenCVSIFTLBPExtractor::~OpenCVSIFTLBPExtractor() {
}

void OpenCVSIFTLBPExtractor::create_sift_lpb_dictionary(
        const vector<Mat>& images, OpenCVSIFTDescExtractor siftExtractor,
        OpenCVLBPDescExtractor lbpExtractor, Mat& dst) {
    for (int i = 0; i < images.size(); i++) {
        Mat img = images[i];
        Mat siftDst;
        vector<KeyPoint> keyPoints;
        siftExtractor.getSIFTDescriptor(img, siftDst, keyPoints);

        Mat grayIm;
        cvtColor(img, grayIm, CV_BGR2GRAY);
        Mat lbpDst;
        lbpExtractor.getOLBPDescriptor(grayIm, lbpDst, keyPoints);

        Mat partialSiftLbp;
        hconcat(siftDst, lbpDst, partialSiftLbp);
        cout << "Combined image SIFT-LBP: " << i << endl;
        dst.push_back(partialSiftLbp);
    }
}

void OpenCVSIFTLBPExtractor::process_sift_lbp_images(const vector<Mat>& images,
        const Mat& dict, Mat& dst) {
    OpenCVSIFTDescExtractor siftExtractor;
    OpenCVLBPDescExtractor lbpExtractor;
    for (int i = 0; i < images.size(); i++) {
        Mat sift_descriptors;
        vector<KeyPoint> siftKeyPoints;
        siftExtractor.getSIFTDescriptor(images[i], sift_descriptors, siftKeyPoints);
        Mat lbp_descriptors;
        //convert image to gray scale
        Mat grayImg;
        cvtColor(images[i], grayImg, CV_BGR2GRAY);
        lbpExtractor.getOLBPDescriptor(grayImg, lbp_descriptors, siftKeyPoints);

        //concat two matrices
        Mat siftLbpDecriptors;
        hconcat(sift_descriptors, lbp_descriptors, siftLbpDecriptors);

        //create a nearest neighbor matcher
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match(siftLbpDecriptors, dict, matches);

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

void OpenCVSIFTLBPExtractor::create_sift_lpb_color_dictionary(
        const vector<Mat>& images, OpenCVSIFTDescExtractor siftExtractor,
        OpenCVLBPDescExtractor lbpExtractor,
        ColorFeatureExtractor colorExtractor, Mat& dst) {
    for (int i = 0; i < images.size(); i++) {
        Mat img = images[i];
        Mat siftDst;
        vector<KeyPoint> keyPoints;
        siftExtractor.getSIFTDescriptor(img, siftDst, keyPoints);

        Mat grayIm;
        cvtColor(img, grayIm, CV_BGR2GRAY);
        Mat lbpDst;
        lbpExtractor.getOLBPDescriptor(grayIm, lbpDst, keyPoints);

        Mat partialSiftLbp;
        hconcat(siftDst, lbpDst, partialSiftLbp);

        Mat colorDst;
        colorExtractor.getPatchColorDescriptors(img, keyPoints, colorDst);

        Mat partialSiftLbpColor;
        hconcat(partialSiftLbp, colorDst, partialSiftLbpColor);

        cout << "Combined image SIFT-LBP-COLOR: " << i << endl;
        dst.push_back(partialSiftLbpColor);
    }
}

void OpenCVSIFTLBPExtractor::process_sift_lbp_color_images(
        const vector<Mat>& images, const Mat& dict, Mat& dst) {
    OpenCVSIFTDescExtractor siftExtractor;
    OpenCVLBPDescExtractor lbpExtractor;
    ColorFeatureExtractor colorExtractor;
    for (int i = 0; i < images.size(); i++) {
        Mat sift_descriptors;
        vector<KeyPoint> siftKeyPoints;
        siftExtractor.getSIFTDescriptor(images[i], sift_descriptors, siftKeyPoints);
        Mat lbp_descriptors;
        //convert image to gray scale
        Mat grayImg;
        cvtColor(images[i], grayImg, CV_BGR2GRAY);
        lbpExtractor.getOLBPDescriptor(grayImg, lbp_descriptors, siftKeyPoints);

        //concat two matrices
        Mat siftLbpDecriptors;
        hconcat(sift_descriptors, lbp_descriptors, siftLbpDecriptors);

        //get color, concat
        Mat color_descriptors;
        colorExtractor.getPatchColorDescriptors(images[i], siftKeyPoints,
                color_descriptors);

        Mat siftLbpColorDescriptors;
        hconcat(siftLbpDecriptors, color_descriptors, siftLbpColorDescriptors);

        //create a nearest neighbor matcher
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match(siftLbpColorDescriptors, dict, matches);

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