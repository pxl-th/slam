#ifndef FRAME_H
#define FRAME_H

#pragma warning(push, 0)
#include<vector>

#include<opencv2/core.hpp>
#pragma warning(pop)

#include"detector.hpp"

namespace slam {

/**
 * Frame class that contains functions for holding and retrieving
 * info (e.g. keypoints) from the frame.
 */
class Frame {
public:
    Detector& detector;

    cv::Mat image, descriptors;
    cv::Mat cameraMatrix, distortions;

    std::vector<cv::KeyPoint> keypoints, undistortedKeypoints;
    double timestamp;
public:
    /**
     * Create frame from given image.
     *
     * @param image Image of the current frame.
     * @param timestamp Timestamp of the image.
     * @param detector Keypoints detector for feature extraction.
     * @param cameraMatrix Camera matrix calculated using Calibration.
     * @param distortions Distortions coefficients calculated using Calibration.
     */
    Frame(
        cv::Mat& image, const double& timestamp, Detector& detector,
        cv::Mat& cameraMatrix, cv::Mat& distortions
    );
private:
    void _undistortKeyPoints();
};

};

#endif
