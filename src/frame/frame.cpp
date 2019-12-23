#pragma warning(push, 0)
#include<algorithm>
#include<iostream>

#include<opencv2/core.hpp>
#include<opencv2/core/types.hpp>
#include<opencv2/calib3d.hpp>
#pragma warning(pop)

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

    // Calculate scales for each level in detector's pyramid.
    // This will be used in Bundle Adjustment as edge's information.
    const double scale = detector.getScaleFactor();
    sigma.push_back(1.0f);
    invSigma.push_back(1.0f);
    scales.push_back(1.0f);

    for (int i = 1; i < detector.getLevels(); i++) {
        scales.push_back(scales[i - 1] * scale);
        sigma.push_back(scales[i] * scales[i]);
        invSigma.push_back(1.0f / sigma[i]);
    }
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
