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

class Evaluator {
public:
    Evaluator();
    virtual ~Evaluator();
    void evaluate(std::string train_categories, std::string test_categories,
            std::string imgdir,
            std::string dict_file, std::string feature_type,
            std::string model_file, const std::map<std::string, int>& categories,
            std::string output_predictions);
    void evaluateUsingKNN(std::string sample_file,
            std::string img, std::string dict_file,
            std::string feature_type, std::string training_set,
            const std::map<std::string, int>& categories);
    
private:
    void getLabelsForKnn(std::string sample_file, 
            const std::map<std::string, int>& categories, 
            cv::Mat& labels);
};

#endif	/* EVALUATOR_H */

