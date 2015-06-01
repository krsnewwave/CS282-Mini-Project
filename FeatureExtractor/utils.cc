
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <map>
#include <boost/algorithm/string.hpp>

inline std::vector<std::string> &split(const std::string &s, char delim,
        std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

inline void categoryMap(std::string sample_file_name,
        std::map<std::string, int>& categories) {
    std::ifstream file(sample_file_name);
    std::string cat;
    int i = 0;
    while (getline(file, cat)) {
        boost::algorithm::trim(cat);
        categories.insert(std::make_pair(cat,++i));
    }
}

template <typename M, typename V>
void map_values_to_vec(const M & m, V & v) {
    for (typename M::const_iterator it = m.begin(); it != m.end(); ++it) {
        v.push_back(it->second);
    }
}

inline std::string get_associated_key(std::map<std::string, int> categories,
        int predicted) {
    for (std::map<std::string, int>::const_iterator it = categories.begin();
            it != categories.end(); ++it) {
        // Repeat if you also want to iterate through the second map.
        if (it->second == static_cast<int> (predicted)) {
            return it->first;
        }
    }
    return "N/A";
}

