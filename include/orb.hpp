#include<vector>

#include<opencv2/features2d.hpp>

namespace slam {

class FeatureDetector{
private:
    cv::Ptr<cv::FeatureDetector> detector;

public:
    FeatureDetector(cv::Ptr<cv::FeatureDetector> detector);
    ~FeatureDetector() = default;

    /* Detect keypoints and calculate their descriptors for given image.
     *
     * Results are written into `keypoints` and `descriptors` arguments.
     * Where `descriptors` will be of [32, N] size, where N is the
     * number of keypoints to detect, specified in constructor.
     * Which may slightly deviate in case of FAST score.
     */
    void detect(
        const cv::InputArray& image,
        std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors
    );
};

};
