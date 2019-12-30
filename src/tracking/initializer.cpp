#pragma warning(push, 0)
#include<iostream>

#include<opencv2/core.hpp>
#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include"converter.hpp"
#include"tracking/initializer.hpp"
#include"tracking/optimizer.hpp"

namespace slam {

Initializer::Initializer(std::shared_ptr<Frame> reference) : reference(reference) {}

std::tuple<cv::Mat, cv::Mat, cv::Mat, std::vector<cv::Point3f>>
Initializer::initialize(std::shared_ptr<Frame> current, std::vector<cv::DMatch>& matches) {
    std::vector<cv::Point2f> referencePoints, currentPoints;
    referencePoints.resize(matches.size());
    currentPoints.resize(matches.size());
    // Copy points from keypoints.
    for (size_t i = 0; i < matches.size(); i++) {
        referencePoints[i] = reference->undistortedKeypoints[matches[i].queryIdx].pt;
        currentPoints[i] = current->undistortedKeypoints[matches[i].trainIdx].pt;
    }

    cv::Mat mask, essential = cv::findEssentialMat(
        referencePoints, currentPoints, *reference->cameraMatrix,
        cv::RANSAC, 0.999, 1.0, mask
    );
    cv::Mat rotation, translation, inliersMask;
    cv::recoverPose(
        essential, referencePoints, currentPoints,
        *reference->cameraMatrix, rotation, translation, inliersMask
    );
    // Complete outliers mask and remove outlier points.
    {
    std::vector<cv::Point2f> rp, cp;
    for (int i = 0; i < inliersMask.rows; i++) {
        if (inliersMask.at<uchar>(i) == 0 || mask.at<uchar>(i) == 0) {
            if (inliersMask.at<uchar>(i) == 0)
                mask.at<uchar>(i) = inliersMask.at<uchar>(i);
            continue;
        }
        rp.push_back(referencePoints[i]);
        cp.push_back(currentPoints[i]);
    }
    referencePoints = rp; currentPoints = cp;
    }

    cv::Mat firstProjection(3, 4, CV_32F, cv::Scalar(0));
    cv::Mat secondProjection(3, 4, CV_32F, cv::Scalar(0));

    // Calculate projection matrices.
    reference->cameraMatrix->copyTo(firstProjection.rowRange(0, 3).colRange(0, 3));
    rotation.copyTo(secondProjection.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(secondProjection.rowRange(0, 3).col(3));
    secondProjection = *reference->cameraMatrix * secondProjection;

    // Reconstruct points.
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
        *reference->cameraMatrix, *reference->distortions
    );
    std::cout << "Reprojection error " << error << std::endl;
    return {rotation, translation, mask, reconstructedPoints};
}

std::shared_ptr<Map> Initializer::initializeMap(
    const std::shared_ptr<Frame> current, const cv::Mat& rotation, const cv::Mat& translation,
    const std::vector<cv::Point3f>& reconstructedPoints,
    const std::vector<cv::DMatch>& matches, const cv::Mat& outliersMask
) {
    auto map = std::make_shared<Map>();

    // Construct transformation matrix out of rotation and translation.
    cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
    rotation.copyTo(pose.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(pose.rowRange(0, 3).col(3));

    auto referenceKeyFrame = std::make_shared<KeyFrame>(
        reference, cv::Mat::eye(4, 4, CV_32F)
    );
    auto currentKeyFrame = std::make_shared<KeyFrame>(current, pose);

    // reference -- query, current -- train
    map->addKeyframe(referenceKeyFrame);
    map->addKeyframe(currentKeyFrame);

    // Add observations to mappoints and add mappoints to map.
    for (size_t i = 0, j = 0; i < matches.size(); i++) {
        if (outliersMask.at<uchar>(static_cast<int>(i)) == 0) continue;

        auto mappoint = std::make_shared<MapPoint>(
            reconstructedPoints[j++], currentKeyFrame
        );
        mappoint->addObservation(referenceKeyFrame, matches[i].queryIdx);
        mappoint->addObservation(currentKeyFrame, matches[i].trainIdx);

        referenceKeyFrame->addMapPoint(matches[i].queryIdx, mappoint);
        currentKeyFrame->addMapPoint(matches[i].trainIdx, mappoint);

        map->addMappoint(mappoint);
    }

    optimizer::globalBundleAdjustment(map, 20);

    float inverseMedianDepth = 1.0f / referenceKeyFrame->medianDepth();
    // TODO: assert that depth is positive

    // Scale translation by inverse median depth.
    auto currentPose = currentKeyFrame->getPose();
    currentPose.col(3).rowRange(0, 3) = (
        currentPose.col(3).rowRange(0, 3) * inverseMedianDepth
    );
    currentKeyFrame->setPose(currentPose);
    // Scale mappoints by inverse median depth.
    for (auto& [id, p] : referenceKeyFrame->getMapPoints())
        p->setWorldPos(p->getWorldPos() * inverseMedianDepth);

    return map;
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

    float score = 0;
    cv::Point2f d;
    for (size_t i = 0; i < imagePoints.size(); i++) {
        d = imagePoints[i] - reprojectedPoints[i];
        score += std::sqrt(d.x * d.x + d.y * d.y);
    }
    return score / imagePoints.size();
}

};
