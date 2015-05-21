/* 
 * File:   Sampler.cc
 * Author: dylan
 * 
 * Created on May 13, 2015, 1:33 PM
 */

#include "feature_extractor.h"
#include <set>

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
        //continue if file does not exist (useful for a single sample file)
        ifstream f(file_name);
        if (!f.good()) {
            continue;
        }
        cout << "Reading: " << file_name << endl;
        Mat img = imread(file_name, CV_LOAD_IMAGE_COLOR);
        //convert to ycrcb
        //        Mat result = siftExtractor.equalizeUsingYCBCR(img);
        //get their color and gray sift features
        Mat colorSift;
        vector<KeyPoint> keyPoints;
        siftExtractor.getSIFTDescriptor(img, colorSift, keyPoints);
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

Mat FeatureExtractor::create_training_descriptors(Dataset ds, Mat dict,
        const map<string, int>& categoryMap, vector<int>& classes) {
    Mat training_features;
    cout << "Constructing Training Descriptors..... " << endl << endl;
    //prepare the sift extractor
    OpenCVSIFTDescExtractor siftExtractor;
    vector<Mat> images;
    for (int i = 0; i < ds.size(); i++) {
        string file = get<1>(ds[i]);
        //continue if file does not exist (useful for a single sample file)
        ifstream f(file);
        if (!f.good()) {
            continue;
        }
        Mat img = imread(file, CV_LOAD_IMAGE_COLOR);
        //        Mat result = siftExtractor.equalizeUsingYCBCR(img);
        images.push_back(img);
        //assign to category -> number map
        string cat = get<0>(ds[i]);
        auto search = categoryMap.find(cat);
        if (search == categoryMap.end()) {
            throw new runtime_error("Category string not recognized.");
        }
        classes.push_back(search->second);
    }
    siftExtractor.process_images(images, dict, training_features);

    cout << "Training features size: " << training_features.size() << endl;
    return training_features;
}

Ptr<FaceRecognizer> FeatureExtractor::trainLBPModel(string sample_file,
        string imgdir, const map<string, int>& categoryMap, vector<int>& classes) {
    cout << "Training LBP model" << endl;
    OpenCVLBPDescExtractor lbpModel;
    //sample all images
    Sampler sampler;
    Dataset ds = sampler.getDataset(sample_file, imgdir);
    vector<Mat> images;
    for (int i = 0; i < ds.size(); i++) {
        //read images
        string file_name = get<1>(ds[i]);
        Mat img = imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
        //equalize
        Mat imgEq;
        equalizeHist(img, imgEq);
        images.push_back(imgEq);
        //assign to category -> number map
        string cat = get<0>(ds[i]);
        auto search = categoryMap.find(cat);
        if (search == categoryMap.end()) {
            throw new runtime_error("Category string not recognized.");
        }
        classes.push_back(search->second);
    }

    return lbpModel.trainLBP(images, classes);
}

/** @function main 
 arguments [sample_file] [img_dir] [feature type]
 *  [create-dictionary] [categories] [output-training-descriptors] [output-model]
 */
int main(int argc, char** argv) {
    FeatureExtractor fe;

    if (argc == 1) {
        cout << "Usage: [sample_file] [img_dir] [feature type] "
                "[create-dictionary] [categories] [output-training-descriptors]"
                " [output-model]" << endl;
        return 0;
    }

    string sample_file = argv[1];
    string imgdir = argv[2];
    string feature_type = argv[3];
    string dictionary_file_name = argv[4];
    string category_file_name = argv[5];
    string output_training_desc = argv[6];
    string output_model = argv[7];

    /*--
     * Loading category file
     * --
     */
    map<string, int> categories;
    categoryMap(category_file_name, categories);

    std::clock_t start2;

    /*--
     * Algorithm for extracting SIFT-only vocabulary and training set
     * --
     */
    if (feature_type == "sift") {
        cout << "SIFT Extraction" << endl;
        //construct dictionary, write to file, if dictionary not available
        ifstream dict_file(dictionary_file_name);
        Mat dict;
        if (!dict_file.good()) {
            start2 = std::clock();
            Mat features = fe.extract_features(sample_file, imgdir);
            dict = fe.create_dictionary(features);
            printf("Dictionary creation lasted: %.3f ms\r\n", (std::clock() - start2) / 1000.0);
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
        vector<int> classes;
        start2 = std::clock();
        training_features = fe.create_training_descriptors(training, dict,
                categories, classes);
        //store the training features
        FileStorage fs2(output_training_desc, FileStorage::WRITE);
        fs2 << "training_set" << training_features;
        fs2.release();

        //copy contents of vector to float array
        printf("Training set extraction lasted : %.3f ms\r\n", (std::clock() - start2) / 1000.0);

    } else if (feature_type == "lbp") {
        vector<int> classes;
        Ptr<FaceRecognizer> faceRecognizer =
                fe.trainLBPModel(sample_file, imgdir, categories, classes);
        //store the model to the output
        faceRecognizer->save(output_training_desc);
        //store the training dataset
        Mat dst = faceRecognizer->getMat("histograms");
        //store the training features
        FileStorage fs2(output_training_desc, FileStorage::WRITE);
        fs2 << "training_set" << dst;
        fs2.release();
    }
    return 0;
}
