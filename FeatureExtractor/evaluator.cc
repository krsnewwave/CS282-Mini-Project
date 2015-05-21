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
#include <opencv2/ml/ml.hpp>
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
        fs["training_set"] >> trainingData;
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
        Ptr<FaceRecognizer> faceRecognizer;
        faceRecognizer->load(training_set);
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
        const map<string, int>& categories, string output_predictions) {
    if (feature_type == "sift") {
        cout << "Batch evaluation: SIFT" << endl;
        //load model
        //load training data

        cout << "Loading training set" << endl;
        Mat trainingData;
        FileStorage fs(training_set_file, FileStorage::READ);
        fs["training_set"] >> trainingData;
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
        //category -> true positives
        map<int, int> tp, predictionCount, classCount;
        int correctAnswers;
        //iterate through the predictions to compute precision
        for (int i = 0; i < features.rows; i++) {
            int pred = static_cast<int> (predictionMat.at<float>(i, 0));
            int truth = static_cast<int> (labels[i]);

            cout << "Prediction: " << pred << "Truth: " << truth << endl;
            if (pred == truth) {
                ++correctAnswers;
                ++tp[pred];
            }
            ++predictionCount[pred];
            ++classCount[truth];
        }

        //compute Recall, Precision, F-Score
        cout << "Recall: " << endl;
        //        map<int, double> precision, recall;
        for (map<int, int>::iterator it = classCount.begin(); it != classCount.end(); ++it) {
            double r = tp[it->first] / it->second;
            cout << "Class: " << it->first << " - " << r << endl;
            //            recall[it->first] = r;
        }
        cout << "Precision" << endl;
        for (map<int, int>::iterator it = predictionCount.begin(); it != predictionCount.end(); ++it) {
            double p = tp[it->first] / it->second;
            cout << "Class: " << it->first << " - " << p << endl;
            //            prediction[it->first] = p;
        }
    }
}

/** @function main 
    @usage [sample file] [imgdir] [dictionary] [feature type] [model file]
 *          [category file name] [output predictions]
 */
int main(int argc, char** argv) {
    if (argc == 1) {
        cout << "Batch classification: usage: ./evaluator "
                "[train categories file] [test categories file] "
                "[imgdir] [dictionary] "
                "[feature type] [model file / training set] "
                "[category file name] [output predictions]" << endl;
        cout << "Single classification: usage: ./evaluator [train categories file] "
                "single [img/dir] [dictionary] "
                "[feature type] [training set / model file] "
                "[category file name]" << endl;
        return 0;
    }

    string train_categories = argv[1];
    string test_categories = argv[2];
    string imgdir_file = argv[3];
    string dict_file = argv[4];
    string feature_type = argv[5];
    string model_file = argv[6];
    string category_file_name = argv[7];

    /*--
     * Loading category file
     * --
     */
    map<string, int> categories;
    categoryMap(category_file_name, categories);

    Evaluator e;
    struct stat sb;
    if (stat(imgdir_file.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        string output_predictions = argv[8];
        e.evaluate(train_categories, test_categories, imgdir_file, dict_file, feature_type, model_file,
                categories, output_predictions);
    } else {
        if (test_categories == "single") {
            e.evaluateUsingKNN(train_categories, imgdir_file, dict_file, feature_type, model_file, categories);
        }
    }
}
