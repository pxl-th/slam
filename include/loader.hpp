#ifndef LOADER_H
#define LOADER_H

#include<opencv2/core.hpp>

namespace slam {

template<typename T>
void save(const T& object, const std::string& file, const std::string& key) {
    cv::FileStorage fs(file, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Could not open file" << file << std::endl;
        return;
    }

    fs << key << object;
    fs.release();
}

template<typename T> std::optional<T>
load(const std::string& file, const std::string& key) {
    cv::FileStorage fs(file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open file" << file << std::endl;
        return {};
    }

    T object;
    fs[key] >> object;
    fs.release();

    return object;
}

};

#endif
