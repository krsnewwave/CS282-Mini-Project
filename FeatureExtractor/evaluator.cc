/* 
 * File:   Evaluator.cpp
 * Author: dylan
 * 
 * Created on May 13, 2015, 5:18 PM
 */

#include "evaluator.h"
#include "sampler.h"
#include "opencv_sift_extractor.h"
#include "feature_extractor.h"
#include <sys/stat.h>

using namespace std;
using namespace cv;

Evaluator::Evaluator() {
}

Evaluator::~Evaluator() {
}

void Evaluator::getLabelsForKnn(string sample_file,
        const map<string, int>& categories,
        Mat& labels) {
    Sampler s;
    Dataset ds = s.getDataset(sample_file, "");
    vector<int> trainingClasses;
    for (int i = 0; i < ds.size(); i++) {
        string cat = get<0>(ds[i]);

        //assign to category -> number map
        auto search = categories.find(cat);
        if (search == categories.end()) {
            throw new runtime_error("Category string not recognized.");
        }
        trainingClasses.push_back(search->second);
    }
    float trng_class_array[trainingClasses.size()]; //copy contents of vector to float array
    copy(trainingClasses.begin(), trainingClasses.end(), trng_class_array);
    labels = Mat(1, trainingClasses.size(), CV_32F, &trng_class_array).clone();
    cout << "Response size: " << labels.size() << endl;
}

void Evaluator::evaluateUsingKNN(string sample_file, string img_file,
        string dict_file,
        string feature_type, string training_set,
        const map<string, int>& categories) {
    if (feature_type == "sift") {
        cout << "Feature type: sift" << endl;
        //load dictionary
        Mat dictionary;
        FileStorage fsDict(dict_file, FileStorage::READ);
        fsDict["vocabulary"] >> dictionary;
        fsDict.release();
        cout << "Dictionary size: " << dictionary.size() << endl;

        //load training data
        //convert training to something that can be read by knn
        cout << "Loading training set" << endl;
        Mat trainingData;
        FileStorage fs(training_set, FileStorage::READ);
        fs["features"] >> trainingData;
        fs.release();
        cout << "Training set size: " << trainingData.size() << endl;

        //prepare the map for training classes -> vector<int>
        Mat trainingClasses_mat;
        getLabelsForKnn(sample_file, categories, trainingClasses_mat);
        //declare KNN
        int K = 32;
        KNearest knn(trainingData, trainingClasses_mat, Mat(), false, 32);

        cout << "Processing test image" << endl;
        //load test image
        OpenCVSIFTDescExtractor ex;
        Mat img = imread(img_file, CV_LOAD_IMAGE_COLOR);
        Mat dst;
        vector<KeyPoint> keyPoints;
        ex.getSIFTDescriptor(img, dst, keyPoints);

        //get nearest neighbor
        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        matcher.match(dst, dictionary, matches);

        float bins[dictionary.rows];

        for (int i = 0; i < matches.size(); i++) {
            bins[matches.at(i).trainIdx] = bins[matches.at(i).trainIdx] + 1;
            //update number of bins
        }

        Mat norm_bins(1, dictionary.rows, CV_32F, &bins);
        normalize(norm_bins, norm_bins, 0, 1, NORM_MINMAX, -1, Mat());
        float predicted = knn.find_nearest(norm_bins, K);
        //get the integer's equivalent in the map
        string cat = get_associated_key(categories,
                static_cast<int> (predicted));
        cout << "Predicted category: " << cat << endl;
    }
    if (feature_type == "lbp") {
        cout << "Loading classifier" << endl;
        Ptr<FaceRecognizer> faceRecognizer = createLBPHFaceRecognizer();
        cout << training_set << endl;
        faceRecognizer->load(training_set);
        cout << "Classifier loaded. Info:" << faceRecognizer->info() << endl;
        Mat img = imread(img_file, CV_LOAD_IMAGE_GRAYSCALE);
        int predicted_label = -1;
        double predicted_confidence = 0.0;
        faceRecognizer->predict(img, predicted_label, predicted_confidence);
        string category = get_associated_key(categories, predicted_label);
        cout << "Predicted: " << category << endl;
    }
}

void Evaluator::evaluate(string train_categories, string test_categories,
        string imgdir, string dict_file,
        string feature_type, string training_set_file,
        const map<string, int>& categories) {
    if (feature_type == "sift") {
        cout << "Batch evaluation: SIFT" << endl;
        //load model
        //load training data

        cout << "Loading training set" << endl;
        Mat trainingData;
        FileStorage fs(training_set_file, FileStorage::READ);
        fs["features"] >> trainingData;
        fs.release();
        cout << "Training set size: " << trainingData.size() << endl;

        //convert training to something that can be read by knn
        //prepare the map for training classes -> vector<int>
        //process test dataset
        Mat training_labels;
        getLabelsForKnn(train_categories, categories, training_labels);

        //declare KNN
        int K = 32;
        KNearest knn(trainingData, training_labels, Mat(), false, 32);

        //load dictionary
        Mat dict;
        FileStorage fs2(dict_file, FileStorage::READ);
        fs2["vocabulary"] >> dict;
        fs2.release();

        //prepare the map for training classes -> vector<int>
        cout << "Preparing test set labels" << endl;
        Mat test_labels;
        getLabelsForKnn(test_categories, categories, test_labels);
        const float* p = (float*) (test_labels.ptr(0));
        vector<float> labels(p, p + test_labels.cols);

        cout << "Extracting test set features" << endl;
        OpenCVSIFTDescExtractor siftExtractor(2000, 3, 0.004);
        //reading all images
        vector<Mat> images;
        Sampler s;
        Dataset testDs = s.getDataset(test_categories, imgdir);
        for (int i = 0; i < testDs.size(); i++) {
            //read images
            string file_name = get<1>(testDs[i]);
            Mat img = imread(file_name, CV_LOAD_IMAGE_COLOR);
            images.push_back(img);
        }

        Mat features;
        siftExtractor.process_images(images, dict, features);
        cout << "Features size: " << features.size() << endl;

        //iterate the test set's features
        Mat predictionMat, n, d;
        knn.find_nearest(features, knn.get_max_k(), predictionMat, n, d);
        cout << "Predictions: [true] [predicted]" << endl;
        //iterate through the predictions
        for (int i = 0; i < features.rows; i++) {
            int pred = static_cast<int> (predictionMat.at<float>(i, 0));
            int truth = static_cast<int> (labels[i]);

            string predicted_class = get_associated_key(categories, pred);
            string true_class = get_associated_key(categories, truth);
            string file_name = get<1>(testDs[i]);

            cout << file_name << "\t" << true_class << "\t" <<
                    predicted_class << endl;
        }
    } else if (feature_type == "lbp") {
        cout << "Loading classifier" << endl;
        Ptr<FaceRecognizer> faceRecognizer = createLBPHFaceRecognizer();
        cout << training_set_file << endl;
        faceRecognizer->load(training_set_file);
        cout << "Classifier loaded. Info:" << faceRecognizer->name() << endl;
        Sampler s;
        Dataset testDs = s.getDataset(test_categories, imgdir);
        cout << "Predictions: [true] [predicted]" << endl;
        for (int i = 0; i < testDs.size(); i++) {
            //read images
            string file_name = get<1>(testDs[i]);
            string true_label = get<0>(testDs[i]);
            Mat img = imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
            Mat imgEq;
            equalizeHist(img, imgEq);
            int predicted_label = -1;
            double predicted_confidence = 0.0;
            faceRecognizer->predict(img, predicted_label, predicted_confidence);
            string predicted_class = get_associated_key(categories, predicted_label);
            cout << file_name << "\t" << true_label << "\t" <<
                    predicted_class << endl;
        }
    }
}

void Evaluator::evaluateSavedFiles(string train_file, string test_file,
        string train_categories, string test_categories,
        const map<string, int>& categories) {
    cout << "YAML evaluation" << endl;
    //load model
    //load training data

    cout << "Loading training set" << endl;
    Mat trainingData;
    FileStorage fs(train_file, FileStorage::READ);
    fs["features"] >> trainingData;
    fs.release();
    cout << "Training set size: " << trainingData.size() << endl;

    //convert training to something that can be read by knn
    //prepare the map for training classes -> vector<int>
    //process test dataset
    Mat training_labels;
    getLabelsForKnn(train_categories, categories, training_labels);

    //declare KNN
    int K = 32;
    KNearest knn(trainingData, training_labels, Mat(), false, 32);

    cout << "Loading test set" << endl;
    Mat testData;
    FileStorage fs3(test_file, FileStorage::READ);
    fs3["features"] >> testData;
    fs3.release();
    cout << "Test set size: " << testData.size() << endl;

    //convert training to something that can be read by knn
    //prepare the map for training classes -> vector<int>
    //process test dataset
    //prepare the map for training classes -> vector<int>
    cout << "Preparing test set labels" << endl;
    Mat test_labels;
    getLabelsForKnn(test_categories, categories, test_labels);
    const float* p = (float*) (test_labels.ptr(0));
    vector<float> labels(p, p + test_labels.cols);
    predict(knn, testData, labels, test_categories, categories);
}

void Evaluator::predict(KNearest& knn, Mat& features,
        const vector<float>& labels, string test_categories,
        const map<string, int> & categories) {
    Sampler s;
    Dataset testDs = s.getDataset(test_categories, "");
    //iterate the test set's features
    Mat predictionMat, n, d;
    knn.find_nearest(features, knn.get_max_k(), predictionMat, n, d);
    cout << "Predictions: [true] [predicted]" << endl;
    //iterate through the predictions
    for (int i = 0; i < features.rows; i++) {
        int pred = static_cast<int> (predictionMat.at<float>(i, 0));
        int truth = static_cast<int> (labels[i]);

        string predicted_class = get_associated_key(categories, pred);
        string true_class = get_associated_key(categories, truth);
        string file_name = get<1>(testDs[i]);

        cout << file_name << "\t" << true_class << "\t" <<
                predicted_class << endl;
    }
}

/** @function main 
    @usage [sample file] [imgdir] [dictionary] [feature type] [model file]
 *          [category file name] [output predictions]
 */
int main(int argc, char** argv) {
    if (argc == 1) {
        cout << "Batch classification: usage: ./evaluator single "
                "[train categories file] [test categories file] "
                "[imgdir] [dictionary] "
                "[feature type] [model file / training set] "
                "[category file name]" << endl;
        cout << "Single classification: usage: ./evaluator single [train set] "
                " [img] [dictionary] feature type] [training set / model file] "
                "[category file name]" << endl;
        cout << "Test your YAML: ./evaluation [train yaml] [test yaml] "
                "[train categories] [test categories] [feature type] "
                "[dictionary] [categories]" << endl;
        return 0;
    }

    /* The variables' names are quite confusing now. Just refer to the above usage
     */
    string mode = argv[1];

    Evaluator e;
    struct stat sb;
    if (mode == "batch") {
        string train_categories = argv[2];
        string test_categories = argv[3];
        string imgdir_file = argv[4];
        string dict_file = argv[5];
        string feature_type = argv[6];
        string model_file = argv[7];
        string category_file_name = argv[8];

        map<string, int> categories;
        categoryMap(category_file_name, categories);
        e.evaluate(train_categories, test_categories, imgdir_file,
                dict_file, feature_type, model_file, categories);
    } else if (mode == "yaml") {
        string train_yaml = argv[2];
        string test_yaml = argv[3];
        string train_sample_file = argv[4];
        string test_sample_file = argv[5];
        string category_file_name = argv[6];
        map<string, int> categories;
        categoryMap(category_file_name, categories);
        e.evaluateSavedFiles(train_yaml, test_yaml, train_sample_file,
                test_sample_file, categories);
    } else if (mode == "single") {
        string train_categories = argv[2];
        string imgdir_file = argv[3];
        string dict_file = argv[4];
        string feature_type = argv[5];
        string model_file = argv[6];
        string category_file_name = argv[7];

        map<string, int> categories;
        categoryMap(category_file_name, categories);
        e.evaluateUsingKNN(train_categories, imgdir_file, dict_file, feature_type, model_file, categories);
    }
}
