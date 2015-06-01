/* 
 * File:   Evaluator.h
 * Author: dylan
 *
 * Created on May 13, 2015, 5:18 PM
 */

#ifndef EVALUATOR_H
#define	EVALUATOR_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <map>
#include "opencv2/core/core.hpp"
#include <opencv2/ml/ml.hpp>

class Evaluator {
public:
    Evaluator();
    virtual ~Evaluator();
    void evaluate(std::string train_categories, std::string test_categories,
            std::string imgdir,
            std::string dict_file, std::string feature_type,
            std::string model_file, const std::map<std::string, int>& categories);
    void evaluateUsingKNN(std::string sample_file,
            std::string img, std::string dict_file,
            std::string feature_type, std::string training_set,
            const std::map<std::string, int>& categories);
    void evaluateSavedFiles(std::string train_file, std::string test_file,
            std::string train_categories, std::string test_categories,
            const std::map<std::string, int>& categories);

private:
    void getLabelsForKnn(std::string sample_file,
            const std::map<std::string, int>& categories,
            cv::Mat& labels);

    void predict(cv::KNearest& knn, cv::Mat& features,
            const std::vector<float>& labels, std::string test_sample,
            const std::map<std::string, int> & categories);
};

#endif	/* EVALUATOR_H */

