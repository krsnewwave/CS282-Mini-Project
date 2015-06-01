/* 
 * File:   Sampler.cc
 * Author: dylan
 * 
 * Created on May 13, 2015, 1:33 PM
 */

#include "feature_extractor.h"
#include "sift_lbp_extractor.h"
#include "opencv_lbp_extractor.h"
#include "opencv_sift_extractor.h"
#include <set>
#include <boost/algorithm/string.hpp>


using namespace std;
using namespace cv;

FeatureExtractor::FeatureExtractor() {

}

FeatureExtractor::~FeatureExtractor() {

}

//Mat FeatureExtractor::extractLBPFeatures(string sample_file, string imgdir) {
//    Sampler s;
//    Dataset training = s.getDataset(sample_file, imgdir);
//    vector<Mat> images;
//    for (int i = 0; i < training.size(); i++) {
//        string file = get<1>(training[i]);
//        //read to gray scale
//        cout << "Reading: " << file << endl;
//        Mat img = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//        Mat imgEq;
//        equalizeHist(img, imgEq);
//        images.push_back(imgEq);
//    }
//    OpenCVLBPDescExtractor lbpExtractor;
//    Mat lbp_features;
//    
//    lbpExtractor.getOLBPDescriptorsFromImages(images, lbp_features);
//    return lbp_features;
//}
//
//Mat FeatureExtractor::extractLBPFeatures(const vector<Mat>& images) {
//    Mat resolvedImages;
//    for (int i = 0; i < images.size(); i++) {
//        Mat grayIm;
//        cvtColor(images[i], grayIm, CV_BGR2GRAY);
//        Mat imgEq;
//        equalizeHist(grayIm, imgEq);
//        resolvedImages.push_back(imgEq);
//    }
//    OpenCVLBPDescExtractor lbpExtractor;
//    Mat lbp_features;
//    lbpExtractor.getOLBPDescriptorsFromImages(resolvedImages, lbp_features);
//    return lbp_features;
//}

Mat FeatureExtractor::extractSIFTFeatures(const vector<Mat>& images) {
    Mat siftFeatures;
    OpenCVSIFTDescExtractor siftExtractor;
    for (int i = 0; i < images.size(); i++) {
        Mat img = images[i];
        Mat dst;
        vector<KeyPoint> keyPoints;
        siftExtractor.getSIFTDescriptor(img, dst, keyPoints);
        siftFeatures.push_back(dst);
    }

    return siftFeatures;
}

Mat FeatureExtractor::extractSIFTFeatures(string sample_file, string imgdir) {
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

Mat FeatureExtractor::createSiftLBPTrainingDescriptors(Dataset ds, Mat dict,
        const map<string, int>& categoryMap, vector<int>& classes) {
    Mat trainingFeatures;
    cout << "Constructing training descriptors" << endl << endl;
    OpenCVSIFTLBPExtractor siftLbpExtractor;
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
    siftLbpExtractor.process_sift_lbp_images(images, dict, trainingFeatures);

    return trainingFeatures;
}

Mat FeatureExtractor::createSiftTrainingDescriptors(Dataset ds, Mat dict,
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
        boost::algorithm::trim(file_name);
        cout << "Reading " << file_name << endl;
        Mat img = imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);

        Mat resz;
        if (getSize().area() == 0) {
            resz = img;
        } else {
            resize(img, resz, getSize(), 0, 0, CV_INTER_AREA);
        }
        //equalize
        Mat imgEq;
        equalizeHist(resz, imgEq);
        images.push_back(imgEq);
        //assign to category -> number map
        string cat = get<0>(ds[i]);
        auto search = categoryMap.find(cat);
        if (search == categoryMap.end()) {

            cout << "Category string not recognized" << endl;
            throw new runtime_error("Category string not recognized.");
        }
        classes.push_back(search->second);
    }

    return lbpModel.trainLBP(images, classes);
}

Size FeatureExtractor::getSize() {
    return FeatureExtractor::resizeSize;
}

void FeatureExtractor::setSize(Size size) {
    FeatureExtractor::resizeSize = size;
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
    if (argc == 9) {
        vector<string> strs;
        boost::split(strs, argv[8], boost::is_any_of("x"));
        int x = atoi(strs[0].c_str());
        int y = atoi(strs[1].c_str());
        cout << "Resize option found: " << x << "x" << y << endl;
        fe.setSize(Size(x, y));
    }

    /*--
     * Loading category file
     * --
     */
    map<string, int> categories;
    categoryMap(category_file_name, categories);

    clock_t start2;

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
            start2 = clock();
            Mat features = fe.extractSIFTFeatures(sample_file, imgdir);
            dict = fe.create_dictionary(features);
            printf("Dictionary creation lasted: %.3f ms\r\n", (clock() - start2) / 1000.0);
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
        start2 = clock();
        training_features = fe.createSiftTrainingDescriptors(training, dict,
                categories, classes);
        //store the training features
        FileStorage fs2(output_training_desc, FileStorage::WRITE);
        fs2 << "features" << training_features;
        fs2.release();

        //copy contents of vector to float array
        printf("Training set extraction lasted : %.3f ms\r\n", (clock() - start2) / 1000.0);

    } else if (feature_type == "lbp") {
        vector<int> classes;
        Ptr<FaceRecognizer> faceRecognizer =
                fe.trainLBPModel(sample_file, imgdir, categories, classes);
        //store the model to the output
        faceRecognizer->save(output_model);
        //store the training dataset
        vector<Mat> dst = faceRecognizer->getMatVector("histograms");
        //store the training features
        if (output_training_desc != "n/a") {
            FileStorage fs2(output_training_desc, FileStorage::WRITE);
            fs2 << "features" << dst;
            fs2.release();
        }
    } else if (feature_type == "sift-lbp") {
        cout << "SIFT-LBP mode" << endl;
        ifstream dict_file(dictionary_file_name);
        Mat dict;
        if (!dict_file.good()) {
            cout << "Creating dictionary " << endl;
            start2 = clock();
            //read images
            Sampler s;
            Dataset vocabDataset = s.getDataset(sample_file, imgdir);
            vector<Mat> images;
            for (int i = 0; i < vocabDataset.size(); i++) {
                Mat img = imread(get<1>(vocabDataset[i]), CV_LOAD_IMAGE_COLOR);
                images.push_back(img);
            }
            OpenCVSIFTLBPExtractor extractor;
            OpenCVSIFTDescExtractor siftExtractor;
            OpenCVLBPDescExtractor lbpExtractor;
            Mat siftLbpFeatures;
            extractor.create_sift_lpb_dictionary(images,
                    siftExtractor, lbpExtractor, siftLbpFeatures);

            dict = fe.create_dictionary(siftLbpFeatures);
            printf("Dictionary creation lasted: %.3f ms\r\n", (clock() - start2) / 1000.0);
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
        Sampler sampler;
        Dataset training = sampler.getDataset(sample_file, imgdir);
        start2 = clock();
        vector<Mat> images;
        for (int i = 0; i < training.size(); i++) {
            string file = get<1>(training[i]);
            Mat img = imread(file, CV_LOAD_IMAGE_COLOR);
            images.push_back(img);
            cout << "Reading file: " << file << endl;
        }
        //create training set
        OpenCVSIFTLBPExtractor siftLbpExtractor;
        Mat siftLbpTrainingSet;
        siftLbpExtractor.process_sift_lbp_images(images, dict, siftLbpTrainingSet);
        cout << "Training set size: " << siftLbpTrainingSet.size() << endl;

        //store the training features
        FileStorage fs2(output_training_desc, FileStorage::WRITE);
        fs2 << "features" << siftLbpTrainingSet;
        fs2.release();

        //copy contents of vector to float array
        printf("Training set extraction lasted : %.3f ms\r\n", (clock() - start2) / 1000.0);
    } else if (feature_type == "sift-lbp-color") {
        cout << "SIFT-LBP-Color mode" << endl;
        ifstream dict_file(dictionary_file_name);
        Mat dict;
        if (!dict_file.good()) {
            cout << "Creating dictionary " << endl;
            start2 = clock();
            //read images
            Sampler s;
            Dataset vocabDataset = s.getDataset(sample_file, imgdir);
            vector<Mat> images;
            for (int i = 0; i < vocabDataset.size(); i++) {
                Mat img = imread(get<1>(vocabDataset[i]), CV_LOAD_IMAGE_COLOR);
                images.push_back(img);
            }
            OpenCVSIFTLBPExtractor extractor;
            OpenCVSIFTDescExtractor siftExtractor;
            ColorFeatureExtractor colorExtractor;
            OpenCVLBPDescExtractor lbpExtractor;

            Mat siftLbpColorDesc;
            extractor.create_sift_lpb_color_dictionary(images, siftExtractor,
                    lbpExtractor, colorExtractor, siftLbpColorDesc);

            dict = fe.create_dictionary(siftLbpColorDesc);
            printf("Dictionary creation lasted: %.3f ms\r\n", (clock() - start2) / 1000.0);
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
        Sampler sampler;
        Dataset training = sampler.getDataset(sample_file, imgdir);
        start2 = clock();
        vector<Mat> images;
        for (int i = 0; i < training.size(); i++) {
            string file = get<1>(training[i]);
            Mat img = imread(file, CV_LOAD_IMAGE_COLOR);
            images.push_back(img);
            cout << "Reading file: " << file << endl;
        }
        //create training set
        OpenCVSIFTLBPExtractor siftLbpExtractor;
        Mat siftLbpTrainingSet;
        siftLbpExtractor.process_sift_lbp_color_images(images,
                dict, siftLbpTrainingSet);
        cout << "Training set size: " << siftLbpTrainingSet.size() << endl;

        //store the training features
        FileStorage fs2(output_training_desc, FileStorage::WRITE);
        fs2 << "features" << siftLbpTrainingSet;
        fs2.release();

        //copy contents of vector to float array
        printf("Training set extraction lasted : %.3f ms\r\n", (clock() - start2) / 1000.0);
    }

    return 0;
}
