#include <cv.h>
#include <stdio.h>
#include <legacy/legacy.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

    Mat dictionary;
template <typename _Tp> void orig_lbp(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
            code |= (src.at<_Tp>(i-1,j) > center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
            code |= (src.at<_Tp>(i,j+1) > center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
            code |= (src.at<_Tp>(i+1,j) > center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
            code |= (src.at<_Tp>(i,j-1) > center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

vector<string> readFile(char* fileName, char* imageFolder){

    vector<string> files;

    ifstream source (fileName);

     for(std::string lines; std::getline(source, lines); )  //read stream line by line
{
     std::istringstream in(lines);      //make a stream for the line itself

    string file;
    in >> file; //read the whitespace-separated double

    files.push_back(imageFolder + file);

}

    return files;
}

vector<int> readClass(char* fileName){

    vector<int> files;

    ifstream source (fileName);

     for(std::string lines; std::getline(source, lines); )  //read stream line by line
{
     std::istringstream in(lines);      //make a stream for the line itself

    int file;
    in >> file; //read the whitespace-separated double

    files.push_back(file);

}

    return files;
}

void buildTrainingDescriptors(){

    Mat training_features;



    FileStorage fs("dictionary.yml", FileStorage::READ);
    fs["vocabulary"] >> dictionary;
    fs.release();


    vector<string> filenames = readFile("files.txt", "../CS296/data/training_images/");

for(int i=0; i<filenames.size();i++){
    string file = filenames.at(i);
    Mat img1 = imread(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE );

    Mat img1_equalized;


  cout << "\nProcessing image: " << file << " -  " << img1.rows << "x" << img1.cols << " iteration - " << i << endl;
    img1_equalized = img1.clone();

    equalizeHist( img1, img1_equalized); // histogram equalization
    cout << "\n\n(1) Image histogram equalized ... ";

    Mat img1_sift = img1_equalized.clone();
    Mat sift_descriptors, sift_dest;
    vector<KeyPoint> keypoints;

    //SiftDescriptorExtractor detector_sift;
    //detector_sift.detect(img1_sift, keypoints);
    //detector_sift.compute(img1_sift, keypoints,sift_descriptors);

    SIFT sift(2000,3,0.004);
    sift(img1_equalized, img1_equalized, keypoints, sift_descriptors, false);

    drawKeypoints(img1_sift, keypoints, sift_dest, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cout << "\n\n(2) SIFT Descriptor for image computed:\n";
    cout << "Keypoints detected: " << keypoints.size() << endl;
    cout << "Feature Vector: " << sift_descriptors.size();

    //lbp<unsigned char>(img1_equalized,  dest_lbp, radius_lbp, neighbor_lbp); /// Local Binary Pattern

Mat dest_lbp;
    orig_lbp<unsigned char>(img1_equalized, dest_lbp);
    cout << "\n\n(3) LBP Computed Image: " << dest_lbp.size()  <<endl;

    Mat dest_lbp_out = dest_lbp.clone();
    drawKeypoints(dest_lbp_out, keypoints, dest_lbp_out, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //obtain patch values per keypoint (8x8)

    Mat patch;
    Mat lbp_descriptors;
    int patchSize = 8;

    Size size = Size(patchSize , patchSize); //square patchSize x patchSize region (8x8 patch)

    for(int i=0; i<keypoints.size();i++){

        Point2f center(keypoints.at(i).pt.x,keypoints.at(i).pt.y);
        getRectSubPix(dest_lbp,size,center,patch);
        patch = patch.reshape(0,1); //flatten matrix to 1-D Vector

        lbp_descriptors.push_back(patch);
    }

    cout << "LBP Descriptors computed ... " << endl;
    cout << "LBP Vector: [" << lbp_descriptors.rows << " x" << lbp_descriptors.cols << "]" << endl;

    lbp_descriptors.convertTo(lbp_descriptors,CV_32FC1);

    Mat sift_lbp_features;

    hconcat(sift_descriptors, lbp_descriptors, sift_lbp_features);



   cout << "\nTotal Image feature vector: [ image " <<  " - " << sift_lbp_features.rows << " key points x " << sift_lbp_features.cols << " features]" << endl;



   // total_features.push_back(sift_lbp_features);
  //  cout << "Total features: " << total_features.rows << endl << endl;

    cout << "===================================================================\n\n";

       //prepare BOW descriptor extractor from the dictionary


    //create a nearest neighbor matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( sift_lbp_features, dictionary, matches );

    cout << "Number of matches: " << matches.size() << endl;

    float bins[dictionary.rows] = {0.0};

    for (int i =0; i<matches.size(); i++){

       bins[matches.at(i).trainIdx]= bins[matches.at(i).trainIdx] + 1; //update number of bins

    }


    Mat norm_bins(1,dictionary.rows,CV_32F,&bins);
    normalize( norm_bins, norm_bins, 0, 1, NORM_MINMAX, -1, Mat() );

    training_features.push_back(norm_bins);

}

//store the vocabulary
FileStorage fs2("training_set.yml", FileStorage::WRITE);
fs2 << "training_set" << training_features;
fs2.release();


}


void trainKNN(){

    cout << "(1) Loading Training Set ...\n";

    //read dictionary

    FileStorage fsDict("dictionary.yml", FileStorage::READ);
    fsDict["vocabulary"] >> dictionary;
    fsDict.release();

    //read training data
    Mat trainingData;
    FileStorage fs("training_set.yml", FileStorage::READ);
    fs["training_set"] >> trainingData;
    fs.release();

    Mat test_features;



    cout << "(2) Loading Training Classes\n";

    vector<int> trainingClasses = readClass("training_classes.txt");
    float trng_class_array[trainingClasses.size()]; //copy contents of vector to float array
    std::copy(trainingClasses.begin(), trainingClasses.end(), trng_class_array);
    Mat trainingClasses_mat = Mat(1, trainingClasses.size(), CV_32FC1, &trng_class_array);

    vector<int> testingClasses = readClass("testing_classes.txt");
    float testingClasses_array[testingClasses.size()]; //copy contents of vector to float array
    std::copy(testingClasses.begin(), testingClasses.end(), testingClasses_array);
    Mat testingClasses_mat = Mat(1, testingClasses.size(), CV_32FC1, &testingClasses_array);
    testingClasses_mat = testingClasses_mat.t();

    Mat predicted(testingClasses_mat.rows, 1, CV_32F);

    int K = 1;

    cout << "(3) Initializing knn classifier \n";
    cv::KNearest knn(trainingData, trainingClasses_mat.t(), cv::Mat(), false, K);

///==============================================================================================

        vector<string> filenames = readFile("test_files.txt", "../CS296/data/testing_images/");

for(int i=0; i<filenames.size();i++){
    string file = filenames.at(1);
    Mat img1 = imread(file.c_str(), CV_LOAD_IMAGE_GRAYSCALE );

    Mat img1_equalized;


    cout << "\nProcessing image: " << file << " -  " << img1.rows << "x" << img1.cols << " iteration - " << i << endl;
    img1_equalized = img1.clone();

    equalizeHist( img1, img1_equalized); // histogram equalization
    cout << "\n\n(1) Image histogram equalized ... ";

    Mat img1_sift = img1_equalized.clone();
    Mat sift_descriptors, sift_dest;
    vector<KeyPoint> keypoints;

    //SiftDescriptorExtractor detector_sift;
    //detector_sift.detect(img1_sift, keypoints);
    //detector_sift.compute(img1_sift, keypoints,sift_descriptors);

    SIFT sift(50,3,0.004);
    sift(img1_equalized, img1_equalized, keypoints, sift_descriptors, false);

    drawKeypoints(img1_sift, keypoints, sift_dest, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cout << "\n\n(2) SIFT Descriptor for image computed:\n";
    cout << "Keypoints detected: " << keypoints.size() << endl;
    cout << "Feature Vector: " << sift_descriptors.size();

    //lbp<unsigned char>(img1_equalized,  dest_lbp, radius_lbp, neighbor_lbp); /// Local Binary Pattern

    Mat dest_lbp;
    orig_lbp<unsigned char>(img1_equalized, dest_lbp);
    cout << "\n\n(3) LBP Computed Image: " << dest_lbp.size()  <<endl;

    Mat dest_lbp_out = dest_lbp.clone();
    drawKeypoints(dest_lbp_out, keypoints, dest_lbp_out, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //obtain patch values per keypoint (8x8)

    Mat patch;
    Mat lbp_descriptors;
    int patchSize = 8;

    Size size = Size(patchSize , patchSize); //square patchSize x patchSize region (8x8 patch)

    for(int i=0; i<keypoints.size();i++){

        Point2f center(keypoints.at(i).pt.x,keypoints.at(i).pt.y);
        getRectSubPix(dest_lbp,size,center,patch);
        patch = patch.reshape(0,1); //flatten matrix to 1-D Vector

        lbp_descriptors.push_back(patch);
    }

    cout << "LBP Descriptors computed ... " << endl;
    cout << "LBP Vector: [" << lbp_descriptors.rows << " x" << lbp_descriptors.cols << "]" << endl;

    lbp_descriptors.convertTo(lbp_descriptors,CV_32FC1);

    Mat sift_lbp_features;

    hconcat(sift_descriptors, lbp_descriptors, sift_lbp_features);


//==============================================================================

    //create a nearest neighbor matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;

    matcher.match( sift_lbp_features, dictionary, matches );

    cout << "Number of matches: " << matches.size() << endl;

    float bins[dictionary.rows] = {0.0};

    for (int i =0; i<matches.size(); i++){

       bins[matches.at(i).trainIdx]= bins[matches.at(i).trainIdx] + 1; //update number of bins

    }


    Mat norm_bins(1,dictionary.rows,CV_32F,&bins);
    normalize( norm_bins, norm_bins, 0, 1, NORM_MINMAX, -1, Mat() );

    predicted.at<float>(i,0) = knn.find_nearest(norm_bins, K);

    cout << "\n\n Predicted class: " << predicted.at<float>(i,0) << " - " << testingClasses_mat.at<float>(i,0) <<  endl;
}
//=======================================================================================


//        plot_binary(testData, prediction, "Predictions Backpropagation");
//store the vocabulary
FileStorage fs2("predicted.yml", FileStorage::WRITE);
fs2 << "predicted" << predicted;
fs2.release();
    /* Mat outImage (img1.rows, img1.cols * 2, CV_8UC1);

    Mat c1(outImage, Rect(0, 0, img1.cols, img1.rows));
    Mat c2(outImage, Rect(img1.cols, 0, img1.cols, img1.rows));




    img1.copyTo(c1);
    img1_equalized.copyTo(c2);


    imshow("SIFT", sift_dest);
    imshow("LBP", dest_lbp_out);
    imshow("Input Image", outImage);*/

}

int main()
{
    trainKNN();





/// ================================================================================//
///***** Displaying Functions ******************************************************//
///================================================================================//

   waitKey(0);
   return 0;
}
