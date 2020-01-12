#ifndef INITIALIZER_H
#define INITIALIZER_H

#pragma warning(push, 0)
#include<tuple>
#pragma warning(pop)

#include"frame/frame.hpp"
#include"map/map.hpp"

namespace slam {

class Initializer {
public:
    std::shared_ptr<KeyFrame> reference;
public:
    Initializer(){}
    Initializer(std::shared_ptr<KeyFrame> reference);
    /**
     * Given `current` frame and matches between
     * `reference` and `current` frames, find initial reconstruction.
     *
     * @param current Current frame, that was successfully matched
     * with `reference` frame.
     * @param matches Matches between `reference` and `current` frames.
     *
     * @return rotatiton, translation, inliers mask (0 value is outlier),
     * reconstructed feature points.
     */
    /* std::tuple<std::vector<cv::Point3f>, cv::Mat, cv::Mat> */
    /* initialize(std::shared_ptr<KeyFrame> current, std::vector<cv::DMatch>& matches); */
    std::shared_ptr<Map> initializeMap(
        const std::shared_ptr<KeyFrame> current,
        const cv::Mat& pose,
        const std::vector<cv::Point3f>& reconstructedPoints,
        const std::vector<cv::DMatch>& matches, const cv::Mat& outliersMask
    );
};

};

#endif
