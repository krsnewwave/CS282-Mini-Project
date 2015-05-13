/* 
 * File:   Evaluator.cpp
 * Author: dylan
 * 
 * Created on May 13, 2015, 5:18 PM
 */

#include "evaluator.h"
#include "sampler.h"
#include "opencv_sift_extractor.h"

using namespace std;
using namespace cv;

Evaluator::Evaluator() {
}

Evaluator::~Evaluator() {
}

void Evaluator::evaluate(std::string sample_file, std::string imgdir,
        std::string dict_file, std::string feature_type) {
    Sampler sampler;
    //load test dataset
    Dataset ds = sampler.getDataset(sample_file, imgdir);
    //load dictionary file
    Mat dictionary;
    FileStorage fsDict(dict_file, FileStorage::READ);
    fsDict["vocabulary"] >> dictionary;
    fsDict.release();
    
    
}

/** @function main */
int main(int argc, char** argv) {
    string sample_file = argv[1];
    string imgdir_file = argv[2];
    string dict_file = argv[3];
    string feature_type = argv[4];

    Evaluator e;
    e.evaluate(sample_file, imgdir_file, dict_file, feature_type);
}
