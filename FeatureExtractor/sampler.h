/* 
 * File:   Sampler.h
 * Author: dylan
 *
 * Created on May 13, 2015, 1:33 PM
 */

#ifndef SAMPLER_H
#define	SAMPLER_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <tuple>
#include "utils.cc"

typedef std::vector<std::tuple<std::string, std::string>> Dataset;
typedef std::vector<std::string> TestDataset;

class Sampler {
public:
    Sampler();
    Sampler(const Sampler& orig);
    virtual ~Sampler();
    Dataset getDataset(std::string sample_file_name, std::string imgdir);
    TestDataset getTestDataset(std::string sample_file_name, std::string imgdir);

private:

};

#endif	/* SAMPLER_H */