#ifndef DETECTOR_H
#define DETECTOR_H

#include<vector>

#include<opencv2/features2d.hpp>

namespace slam {

/**
 * Detector class for finding keypoints in the images.
 */
class Detector{
private:
    cv::Ptr<cv::ORB> detector;

public:
    Detector(cv::Ptr<cv::ORB> detector);
    ~Detector() = default;

    /**
     * Detect keypoints and calculate their descriptors for given image.
     *
     * Results are written into `keypoints` and `descriptors` arguments.
     * Where `descriptors` will be of [32, N] size, where N is the
     * number of keypoints to detect, specified in constructor.
     * Which may slightly deviate in case of FAST score.
     */
    void detect(
        cv::InputArray& image,
        std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors
    );

    inline int getLevels() { return detector->getNLevels(); }
    inline double getScaleFactor() { return detector->getScaleFactor(); }
};

};

#endif
