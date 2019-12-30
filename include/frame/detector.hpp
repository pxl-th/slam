#ifndef DETECTOR_H
#define DETECTOR_H

#pragma warning(push, 0)
#include<vector>

#include<opencv2/features2d.hpp>
#pragma warning(pop)

namespace slam {

/**
 * Detector class for finding keypoints in the images.
 */
class Detector{
private:
    cv::Ptr<cv::ORB> detector;
public:
    Detector(){}
    Detector(cv::Ptr<cv::ORB> detector);
    /**
     * Detect keypoints and calculate their descriptors for given image.
     *
     * Results are written into `keypoints` and `descriptors` arguments.
     * Where `descriptors` will be of [32, N] size, where N is the
     * number of keypoints to detect, specified in constructor.
     * Which may slightly deviate in case of FAST score.
     */
    void detect(
        std::shared_ptr<cv::Mat> image,
        std::vector<cv::KeyPoint>& keypoints,
        std::shared_ptr<cv::Mat> descriptors
    );

    inline int getLevels() { return detector->getNLevels(); }
    inline double getScaleFactor() { return detector->getScaleFactor(); }
};

};

#endif
