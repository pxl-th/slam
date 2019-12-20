#include<algorithm>
#include<iostream>

#include<opencv2/core.hpp>
#include<opencv2/core/types.hpp>
#include<opencv2/calib3d.hpp>

#include"frame/frame.hpp"

namespace slam {

Frame::Frame(
    cv::Mat& image, const double& timestamp, Detector& detector,
    cv::Mat& cameraMatrix, cv::Mat& distortions
) : image(image), timestamp(timestamp), detector(detector),
    cameraMatrix(cameraMatrix.clone()), distortions(distortions.clone()) {

    // Detect and extract keypoints and their descriptors.
    detector.detect(image, keypoints, descriptors);
    if (keypoints.empty()) return;
    _undistortKeyPoints();
}

void Frame::_undistortKeyPoints() {
    // If no distortion, then nothing to undistort.
    if (distortions.at<float>(0) == 0.0f) {
        undistortedKeypoints = keypoints;
        return;
    }

    // Convert keypoints to matrix representation.
    cv::Mat undistorted(static_cast<int>(keypoints.size()), 2, CV_64F);
    for (int i = 0; i < keypoints.size(); i++) {
        undistorted.at<float>(i, 0) = keypoints[i].pt.x;
        undistorted.at<float>(i, 1) = keypoints[i].pt.y;
    }

    undistorted.reshape(2);
    cv::undistortPoints(
        undistorted, undistorted, cameraMatrix,
        distortions, cv::Mat(), cameraMatrix
    );
    undistorted.reshape(1);

    // Convert undistorted keypoints from matrix representation to list.
    undistortedKeypoints.resize(keypoints.size());
    for(int i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint p = keypoints[i];
        p.pt.x = undistorted.at<float>(i, 0);
        p.pt.y = undistorted.at<float>(i, 1);
        undistortedKeypoints[i] = p;
    }
}

};
