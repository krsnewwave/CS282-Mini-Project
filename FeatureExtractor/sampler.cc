/* 
 * File:   Sampler.cc
 * Author: dylan
 * 
 * Created on May 13, 2015, 1:33 PM
 */
#include "sampler.h"
#include <sstream>
#include <fstream>

using namespace std;

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
        tuple<string, string> t = make_tuple(line[0], imgdir + "/" + line[1]);
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