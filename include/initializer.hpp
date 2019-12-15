#include<tuple>

#include"frame.hpp"

namespace slam {

class Initializer {
public:
    const Frame& reference;

public:
    Initializer(const Frame& reference);
    ~Initializer() = default;

    std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<cv::Point3f>>
    initialize(const Frame& current, const std::vector<cv::DMatch>& matches);
};

};
