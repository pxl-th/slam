#include<tuple>

#include"frame/frame.hpp"

namespace slam {

class Initializer {
public:
    const Frame& reference;

public:
    Initializer(const Frame& reference);
    ~Initializer() = default;

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
    initialize(const Frame& current, const std::vector<cv::DMatch>& matches);

private:
    float _reprojectionError(
        std::vector<cv::Point2f>& imagePoints,
        std::vector<cv::Point3f>& objectPoints,
        cv::Mat& rotation, cv::Mat& translation,
        const cv::Mat& cameraMatrix, const cv::Mat& distortions
    );
};

};
