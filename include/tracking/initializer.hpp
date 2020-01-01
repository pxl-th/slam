#pragma warning(push, 0)
#include<tuple>
#pragma warning(pop)

#include"frame/frame.hpp"
#include"map/map.hpp"

namespace slam {

class Initializer {
public:
    std::shared_ptr<Frame> reference;
public:
    Initializer(){}
    Initializer(std::shared_ptr<Frame> reference);
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
    std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<cv::Point3f>>
    initialize(std::shared_ptr<Frame> current, std::vector<cv::DMatch>& matches);

    std::shared_ptr<Map> initializeMap(
        const std::shared_ptr<Frame> current,
        const cv::Mat& rotation, const cv::Mat& translation,
        const std::vector<cv::Point3f>& reconstructedPoints,
        const std::vector<cv::DMatch>& matches, const cv::Mat& outliersMask
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
