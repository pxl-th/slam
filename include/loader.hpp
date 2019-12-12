/**
 * Contains functions for serialization and deserialization of objects.
 */
#ifndef LOADER_H
#define LOADER_H

#include<opencv2/core.hpp>

namespace slam {

template<typename T>
void write(cv::FileStorage& fs, const std::string&, const T& object) {
    object.write(fs);
}

template<typename T>
void read(const cv::FileNode& node, T& object, const T& defaultObject) {
    if(node.empty()) object = defaultObject;
    else object.read(node);
}

/**
 * Serialize given object.
 *
 * @param object Object to serialize.
 * @param file Filepath where serialized object will be saved.
 * @param key Key value under which object will be saved.
 * @param append If `True`, then serialized object will be appended
 * to the `file` under given `key`.
 *
 * @throw std::invalid_argument If given file path could not be opened.
 */
template<typename T>
void save(
    const T& object,
    const std::string& file, const std::string& key,
    bool append = false
) {
    cv::FileStorage fs(
        file, append ? cv::FileStorage::APPEND : cv::FileStorage::WRITE
    );
    if (!fs.isOpened()) {
        std::cerr << "Could not open file " << file << std::endl;
        throw std::invalid_argument(std::string("Could not open file ") + file);
    }

    fs << key << object;
    fs.release();
}

/**
 * Deserialize object.
 *
 * @param file Filepath that contains serialized object.
 * @param key Key value under which object was serialized.
 * @return Deserialized object.
 *
 * @throw std::invalid_argument If given file path could not be opened.
 */
template<typename T> T load(const std::string& file, const std::string& key) {
    cv::FileStorage fs(file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open file " << file << std::endl;
        throw std::invalid_argument(std::string("Could not open file ") + file);
    }

    T object;
    fs[key] >> object;
    fs.release();

    return object;
}

};

#endif
