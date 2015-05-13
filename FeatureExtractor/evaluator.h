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

class Evaluator {
public:
    Evaluator();
    virtual ~Evaluator();
    void evaluate(std::string sample_file, std::string imgdir,
            std::string dict_file, std::string feature_type);
};

#endif	/* EVALUATOR_H */

