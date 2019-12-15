#include<iostream>
#include<opencv2/calib3d.hpp>

#include"initializer.hpp"

namespace slam {

Initializer::Initializer(const Frame& reference) : reference(reference) {}

std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<cv::Point3f>>
Initializer::initialize(
    const Frame& current, const std::vector<cv::DMatch>& matches
) {
    std::vector<cv::Point2f> referencePoints, currentPoints;
    referencePoints.resize(matches.size());
    currentPoints.resize(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        referencePoints[i] = reference.undistortedKeypoints[matches[i].queryIdx].pt;
        currentPoints[i] = current.undistortedKeypoints[matches[i].trainIdx].pt;
    }

    cv::Mat essential = cv::findEssentialMat(
        referencePoints, currentPoints, reference.cameraMatrix, cv::RANSAC
    );
    cv::Mat rotation, translation, inliersMask;
    cv::recoverPose(
        essential, referencePoints, currentPoints,
        reference.cameraMatrix, rotation, translation, inliersMask
    );
    std::vector<cv::Point2f> refPoints, curPoints;
    for (size_t i = 0; i < inliersMask.rows; i++) {
        if (inliersMask.at<uchar>(i) == 0) continue;

        refPoints.push_back(referencePoints[i]);
        curPoints.push_back(currentPoints[i]);
    }
    std::cout << refPoints.size() << std::endl;

    cv::Mat firstProjection(3, 4, CV_32F, cv::Scalar(0));
    cv::Mat secondProjection(3, 4, CV_32F, cv::Scalar(0));

    reference.cameraMatrix.copyTo(firstProjection.rowRange(0, 3).colRange(0, 3));
    rotation.copyTo(secondProjection.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(secondProjection.rowRange(0, 3).col(3));
    secondProjection = reference.cameraMatrix * secondProjection;

    cv::Mat homogeneousPoints;
    cv::triangulatePoints(
        firstProjection, secondProjection,
        refPoints, curPoints, homogeneousPoints
    );
    std::vector<cv::Point3f> reconstructedPoints;
    for (size_t i = 0; i < homogeneousPoints.cols; i++) {
        cv::Point3f p(
            homogeneousPoints.at<float>(0, i) / homogeneousPoints.at<float>(3, i),
            homogeneousPoints.at<float>(1, i) / homogeneousPoints.at<float>(3, i),
            homogeneousPoints.at<float>(2, i) / homogeneousPoints.at<float>(3, i)
        );
        reconstructedPoints.push_back(p);
    }

    return {rotation, translation, inliersMask, reconstructedPoints};
}

};
