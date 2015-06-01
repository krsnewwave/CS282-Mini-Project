/* 
 * File:   Sampler.cc
 * Author: dylan
 * 
 * Created on May 13, 2015, 1:33 PM
 */
#include "sampler.h"
#include <sstream>
#include <fstream>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost::algorithm;

Sampler::Sampler() {
}

Sampler::Sampler(const Sampler& orig) {
}

Sampler::~Sampler() {
}

Dataset Sampler::getDataset(string sample_file_name,string imgdir) {
    Dataset ds;
    ifstream file(sample_file_name);
    string str;
    while (getline(file, str)) {
        vector<string> line = split(str, '\t');
        string cat =line[0];
        trim(cat);
        string file_name = imgdir + "/" + line[1];
        trim(file_name);
        tuple<string, string> t = make_tuple(cat, file_name);
        ds.push_back(t);
    }
    return ds;
}

TestDataset Sampler::getTestDataset(std::string sample_file_name,
        string imgdir) {
    TestDataset ds;
    ifstream file(sample_file_name);
    string str;
    while (getline(file, str)) {
        ds.push_back(str + imgdir);
    }
}