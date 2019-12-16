#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/calib3d.hpp>

#include"converter.hpp"
#include"initializer.hpp"

namespace slam {

Initializer::Initializer(const Frame& reference) : reference(reference) {}

std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<cv::Point3f>>
Initializer::initialize(
    const Frame& current, const std::vector<cv::DMatch>& matches
) {
    std::vector<cv::Point2f>
        referencePoints, referencePointTmp,
        currentPoints, currentPointsTmp;

    referencePoints.resize(matches.size());
    currentPoints.resize(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        referencePoints[i] = reference.undistortedKeypoints[matches[i].queryIdx].pt;
        currentPoints[i] = current.undistortedKeypoints[matches[i].trainIdx].pt;
    }

    cv::Mat mask, essential = cv::findEssentialMat(
        referencePoints, currentPoints, reference.cameraMatrix,
        cv::RANSAC, 0.999, 1.0, mask
    );
    for (size_t i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i) == 0) continue;
        referencePointTmp.push_back(referencePoints[i]);
        currentPointsTmp.push_back(currentPoints[i]);
    }
    referencePoints = referencePointTmp;
    currentPoints = currentPointsTmp;

    cv::Mat rotation, translation, inliersMask;
    cv::recoverPose(
        essential, referencePoints, currentPoints,
        reference.cameraMatrix, rotation, translation, inliersMask
    );
    for (size_t i = 0; i < inliersMask.rows; i++) {
        if (inliersMask.at<uchar>(i) == 0) continue;
        referencePointTmp.push_back(referencePoints[i]);
        currentPointsTmp.push_back(currentPoints[i]);
    }
    referencePoints = referencePointTmp;
    currentPoints = currentPointsTmp;

    cv::Mat firstProjection(3, 4, CV_32F, cv::Scalar(0));
    cv::Mat secondProjection(3, 4, CV_32F, cv::Scalar(0));

    reference.cameraMatrix.copyTo(firstProjection.rowRange(0, 3).colRange(0, 3));
    rotation.copyTo(secondProjection.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(secondProjection.rowRange(0, 3).col(3));
    secondProjection = reference.cameraMatrix * secondProjection;

    cv::Mat reconstructedPointsM, homogeneousPoints;
    cv::triangulatePoints(
        firstProjection, secondProjection,
        referencePoints, currentPoints, homogeneousPoints
    );
    homogeneousPoints = homogeneousPoints.t();
    cv::convertPointsFromHomogeneous(homogeneousPoints, reconstructedPointsM);
    auto reconstructedPoints = vectorFromMat(reconstructedPointsM);

    float error = _reprojectionError(
        currentPoints, reconstructedPoints, rotation, translation,
        reference.cameraMatrix, reference.distortions
    );
    std::cout << "Reprojection error " << error << std::endl;

    return {rotation, translation, inliersMask, reconstructedPoints};
}

float Initializer::_reprojectionError(
    std::vector<cv::Point2f>& imagePoints,
    std::vector<cv::Point3f>& objectPoints,
    cv::Mat& rotation, cv::Mat& translation,
    const cv::Mat& cameraMatrix, const cv::Mat& distortions
) {
    cv::Mat rotationVector;
    cv::Rodrigues(rotation, rotationVector);
    std::vector<cv::Point2f> reprojectedPoints;
    cv::projectPoints(
        objectPoints, rotationVector, translation,
        cameraMatrix, distortions,
        reprojectedPoints
    );

    double score = 0;
    cv::Point2f d;
    for (size_t i = 0; i < imagePoints.size(); i++) {
        d = imagePoints[i] - reprojectedPoints[i];
        score += std::sqrt(d.x * d.x + d.y * d.y);
    }
    score /= imagePoints.size();

    return score;
}

};
