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

int radius_lbp = 1;
int neighbor_lbp = 16;
Mat dest_lbp;
Mat img1_equalized;

Mat total_features;


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


template <typename _Tp> void lbp(const Mat& src, Mat& dst, int radius, int neighbors) {
    neighbors = max(min(neighbors,31),1); // set bounds...
    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) && (abs(t-src.at<_Tp>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}


void setRadiusLBP( int radius ) {

  radius_lbp = radius;
  lbp<unsigned char>(img1_equalized,  dest_lbp, radius_lbp, neighbor_lbp); /// Local Binary Pattern

  imshow("LBP", dest_lbp);


}

void setNeighborLBP( int neighbors ) {

  neighbor_lbp = neighbors;

   lbp<unsigned char>(img1_equalized,  dest_lbp, radius_lbp, neighbor_lbp); /// Local Binary Pattern

  imshow("LBP", dest_lbp);


}


vector<string> readFile(char* fileName){

    vector<string> files;

    ifstream source (fileName);

     for(std::string lines; std::getline(source, lines); )  //read stream line by line
{
     std::istringstream in(lines);      //make a stream for the line itself

    string file;
    in >> file; //read the whitespace-separated double

    files.push_back("data/UNICT-FD889/" + file);

}

    return files;
}


void constructDictionary(){



/// ================================================================================//
///***** Bag Word Model Construction ******************************************************//
///================================================================================//

cout << "Constructing Dictionary..... " << endl << endl;


//Construct BOWKMeansTrainer
//the number of bags
int dictionarySize=1000;


cout << "Dictionary Size: " << dictionarySize;

//define Term Criteria
TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
//retries number

int retries=1;

//necessary flags
int flags=KMEANS_PP_CENTERS;

//Create the BoW (or BoF) trainer
BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);

//cluster the feature vectors
Mat dictionary=bowTrainer.cluster(total_features);

//store the vocabulary
FileStorage fs("dictionary.yml", FileStorage::WRITE);
fs << "vocabulary" << dictionary;
fs.release();

}
int main()
{




    cout << "Reading image file names in Data Folder ... \n";
    vector<string> filenames = readFile("file_foods.txt");
    //std::random_shuffle ( filenames.begin(), filenames.end() );

    int n_images = 1000;
//for(int i=0; i<n_images; i++){
    string file = filenames.at(1500);

/// Load image into matrix file in GRAYSCALE

    Mat img1 = imread(file.c_str(), CV_LOAD_IMAGE_COLOR );

    Mat img1_RGB = img1.clone();
imshow("FOODS", img1);

cvtColor(img1,img1,CV_RGB2GRAY);  //  if(! img1.data) continue;

  //  cout << "\nProcessing image: " << file << " - " << img1.rows << "x" << img1.cols << " iteration - " << i << endl;
    img1_equalized = img1.clone();

    equalizeHist( img1, img1_equalized); // histogram equalization
    cout << "\n\n(1) Image histogram equalized ... ";

    Mat img1_sift = img1_equalized.clone();
    Mat sift_descriptors, sift_dest;
    vector<KeyPoint> keypoints;

    SiftDescriptorExtractor detector;
    detector.detect(img1_RGB, keypoints);
    detector.compute(img1_RGB, keypoints,sift_descriptors);

    drawKeypoints(img1_RGB, keypoints, sift_dest, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cout << "\n\n(2) SIFT Descriptor for image computed:\n";
    cout << "Keypoints detected: " << keypoints.size() << endl;
    cout << "Feature Vector: " << sift_descriptors.size();

    //lbp<unsigned char>(img1_equalized,  dest_lbp, radius_lbp, neighbor_lbp); /// Local Binary Pattern

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

        imshow("Parch",patch);
    }


    cout << "LBP Descriptors computed ... " << endl;
    cout << "LBP Vector: [" << lbp_descriptors.rows << " x" << lbp_descriptors.cols << "]" << endl;

    lbp_descriptors.convertTo(lbp_descriptors,CV_32FC1);

    Mat sift_lbp_features;

    hconcat(sift_descriptors, lbp_descriptors, sift_lbp_features);

  //  cout << "\nTotal Image feature vector: [ image " << i << " - " << sift_lbp_features.rows << " key points x " << sift_lbp_features.cols << " features]" << endl;

    total_features.push_back(sift_lbp_features);
    cout << "Total features: " << total_features.rows << endl << endl;

    cout << "===================================================================\n\n";
//}


//constructDictionary();

/// ================================================================================//
///***** Displaying Functions ******************************************************//
///================================================================================//

    Mat outImage (img1.rows, img1.cols * 2, CV_8UC1);

    Mat c1(outImage, Rect(0, 0, img1.cols, img1.rows));
    Mat c2(outImage, Rect(img1.cols, 0, img1.cols, img1.rows));




    img1.copyTo(c1);
    img1_equalized.copyTo(c2);


    imshow("SIFT", sift_dest);
    imshow("LBP", dest_lbp_out);
    imshow("Input Image", outImage);

    cvCreateTrackbar("radius: ","LBP", &radius_lbp, 100, setRadiusLBP);
    cvCreateTrackbar("neighbor: ","LBP", &neighbor_lbp, 100, setNeighborLBP);

    waitKey(0);
    return 0;
}
