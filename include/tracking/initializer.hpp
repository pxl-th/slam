#include<tuple>

#include"frame/frame.hpp"
#include"map/map.hpp"

namespace slam {

class Initializer {
public:
    Frame& reference;

public:
    Initializer(Frame& reference);
    /**
     * Given `current` frame and matches between
     * `reference` and `current` frames, find initial reconstruction.
     *
     * @param current Current frame, that was successfully matched
     * with `reference` frame.
     * @param matches Matches between `reference` and `current` frames.
     *
     * @return rotatiton, translation, inliers mask (0 value is outlier),
     * reconstructed feature points
     */
    std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<cv::Point3f>>
    initialize(Frame& current, std::vector<cv::DMatch>& matches);

    Map initializeMap(
        const Frame& current, const cv::Mat& rotation, const cv::Mat& translation,
        std::vector<cv::Point3f> reconstructedPoints
    );

private:
    float _reprojectionError(
        std::vector<cv::Point2f>& imagePoints,
        std::vector<cv::Point3f>& objectPoints,
        cv::Mat& rotation, cv::Mat& translation,
        const cv::Mat& cameraMatrix, const cv::Mat& distortions
    );
};

};
