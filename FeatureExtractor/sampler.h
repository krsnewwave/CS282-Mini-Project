/* 
 * File:   Sampler.h
 * Author: dylan
 *
 * Created on May 13, 2015, 1:33 PM
 */

#ifndef SAMPLER_H
#define	SAMPLER_H

#endif	/* SAMPLER_H */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <tuple>

using namespace std;

typedef vector<tuple<string, string>> Dataset;
typedef vector<string> TestDataset;

class Sampler {
public:
    Sampler();
    Sampler(const Sampler& orig);
    virtual ~Sampler();
    Dataset getDataset(string sample_file_name, string imgdir);
    TestDataset getTestDataset(std::string sample_file_name, string imgdir);

private:

};

