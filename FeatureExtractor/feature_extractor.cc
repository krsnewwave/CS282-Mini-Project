/* 
 * File:   Sampler.cc
 * Author: dylan
 * 
 * Created on May 13, 2015, 1:33 PM
 */

#include "feature_extractor.h"

using namespace std;
using namespace cv;

FeatureExtractor::FeatureExtractor() {

}

FeatureExtractor::~FeatureExtractor() {

}

Mat FeatureExtractor::extract_features(string sample_file, string imgdir) {
    //get the Sampler
    Sampler sampler;
    Dataset ds = sampler.getDataset(sample_file, imgdir);
    //init SIFT descriptor object
    OpenCVSIFTDescExtractor siftExtractor;
    Mat total_features;

    for (int i = 0; i < ds.size(); i++) {
        //read each file
        string file_name = get<1>(ds[i]);
        Mat img = imread(file_name, CV_LOAD_IMAGE_COLOR);
        //convert to ycrcb
        Mat result = siftExtractor.equalizeUsingYCBCR(img);
        //get their color and gray sift features
        Mat colorSift = siftExtractor.getSIFTDescriptor(result);
        //disregarding gray for now, I can't concat them below
        //        Mat gray;
        //        cvtColor(img, gray, CV_RGB2GRAY);
        //        Mat graySift = siftExtractor.getSIFTDescriptor(gray);

        total_features.push_back(colorSift);
    }
    //    cout << "Total features: " << total_features.size() << endl;
    return total_features;
}

Mat FeatureExtractor::create_dictionary(Mat features) {
    cout << "Constructing Dictionary..... " << endl << endl;


    //Construct BOWKMeansTrainer
    //the number of bags
    int dictionarySize = 1000;

    //define Term Criteria
    TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
    //retries number

    int retries = 1;

    //necessary flags
    int flags = KMEANS_PP_CENTERS;

    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);

    //cluster the feature vectors
    Mat dictionary = bowTrainer.cluster(features);
    cout << "Dictionary size: " << dictionary.size() << endl;
    return dictionary;
}

Mat FeatureExtractor::create_training_descriptors(Dataset ds, Mat dict) {
    Mat training_features;
     cout << "Constructing Training Descriptors..... " << endl << endl;
    //prepare the sift extractor
    OpenCVSIFTDescExtractor siftExtractor(2000, 3, 0.004);
    for (int i = 0; i < ds.size(); i++) {
        string file = get<1>(ds[i]);
        Mat img = imread(file, CV_LOAD_IMAGE_COLOR);
        Mat result = siftExtractor.equalizeUsingYCBCR(img);
        Mat sift_descriptors = siftExtractor.getSIFTDescriptor(result);

        //create a nearest neighbor matcher
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match(sift_descriptors, dict, matches);

        //group to bins
        //        float bins[dict.rows] = {0.0};
        float bins[dict.rows];

        //update number of bins
        for (int i = 0; i < matches.size(); i++) {
            bins[matches.at(i).trainIdx] = bins[matches.at(i).trainIdx] + 1;
        }

        Mat norm_bins(1, dict.rows, CV_32F, &bins);
        normalize(norm_bins, norm_bins, 0, 1, NORM_MINMAX, -1, Mat());
        training_features.push_back(norm_bins);
    }
    
    cout << "Training features size: " << training_features.size() << endl;
    return training_features;
}

/** @function main 
 arguments [sample_file] [img_dir] [feature type]
 *  [create-dictionary] [output-training-descriptors]
 */
int main(int argc, char** argv) {
    FeatureExtractor fe;
    string sample_file = argv[1];
    string imgdir = argv[2];
    string feature_type = argv[3];
    string dictionary_file_name = argv[4];
    string output_training_desc = argv[5];

    /*--
     * Algorithm for extracting SIFT-only vocabulary and training set
     * --
     */
    if (feature_type == "sift") {
        Mat features = fe.extract_features(sample_file, imgdir);
        //construct dictionary, write to file, if dictionary not available
        ifstream dict_file(dictionary_file_name);
        Mat dict;
        if (!dict_file.good()) {
            dict = fe.create_dictionary(features);
            //store the vocabulary
            FileStorage fs(dictionary_file_name, FileStorage::WRITE);
            fs << "vocabulary" << dict;
            fs.release();
        } else {
            //else, read to dict
            FileStorage fs(dictionary_file_name, FileStorage::READ);
            fs["vocabulary"] >> dict;
            fs.release();
        }

        //construct training set descriptors
        Mat training_features;
        Sampler sampler;
        Dataset training = sampler.getDataset(sample_file, imgdir);
        training_features = fe.create_training_descriptors(training, dict);
        //store the training features
        FileStorage fs2(output_training_desc, FileStorage::WRITE);
        fs2 << "training_set" << training_features;
        fs2.release();
    }
    return 0;
}
