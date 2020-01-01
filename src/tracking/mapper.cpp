#pragma warning(push, 0)
#include<opencv2/core.hpp>
#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include"converter.hpp"
#include"tracking/mapper.hpp"

namespace slam {

Mapper::Mapper(std::shared_ptr<Map> map) : map(map), acceptKeyframes(true) {}

bool Mapper::accepts() const { return acceptKeyframes; }

void Mapper::addKeyframe(std::shared_ptr<KeyFrame> keyframe) {
    keyframeQueue.push(keyframe);
}

void Mapper::_processKeyFrame(std::shared_ptr<KeyFrame> keyframe) {
    acceptKeyframes = false;

    currentKeyFrame = keyframeQueue.front();
    keyframeQueue.pop();

    _createConnections(currentKeyFrame);

    map->addKeyframe(keyframe);

    /**
     * + create connections between keyframes before adding to the map
     * + add kf to map
     * + triangulate points
     * - replace triangulation in initializer
     * - for every keyframe connection, triangulate
     * - fuse duplicates
     */
}

void Mapper::_createConnections(
    std::shared_ptr<KeyFrame> targetKeyFrame, int threshold
) {
    targetKeyFrame->connections.clear();
    std::map<std::shared_ptr<KeyFrame>, int> counter;
    // Count number of mappoints that are shared
    // between each KeyFrame and this KeyFrame.
    for (const auto& [id, mappoint] : targetKeyFrame->getMapPoints()) {
        for (const auto& [keyframe, keypointId] : mappoint->getObservations()) {
            if (keyframe->id == targetKeyFrame->id) continue;
            counter[keyframe]++;
        }
    }
    // Create connections with KeyFrames that more than `threshold`
    // shared MapPoints.
    int maxCount = 0;
    for (const auto& [keyframe, count] : counter) {
        if (count > maxCount) maxCount = count;
        if (count < threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
    // If no KeyFrame's have been found --- add KeyFrame with maximum count.
    if (targetKeyFrame->connections.empty()) threshold = maxCount;
    for (const auto& [keyframe, count] : counter) {
        if (count != threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
}

std::tuple<std::vector<cv::Point3f>, cv::Mat, cv::Mat>
Mapper::triangulatePoints(
    std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2,
    std::vector<cv::DMatch> matches
) {
    cv::Mat cameraMatrix = *frame1->cameraMatrix;
    // Copy points from keypoints.
    std::vector<cv::Point2f> frame1Points, frame2Points;
    frame1Points.resize(matches.size());
    frame2Points.resize(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        frame1Points[i] = frame1->undistortedKeypoints[matches[i].queryIdx].pt;
        frame2Points[i] = frame2->undistortedKeypoints[matches[i].trainIdx].pt;
    }
    // Calculate essential matrix and recover pose from it.
    cv::Mat mask, essential = cv::findEssentialMat(
        frame1Points, frame2Points, cameraMatrix,
        cv::RANSAC, 0.999, 1.0, mask
    );
    cv::Mat rotation, translation, inliersMask;
    cv::recoverPose(
        essential, frame1Points, frame2Points,
        cameraMatrix, rotation, translation, inliersMask
    );
    // Complete outliers `mask` and remove outlier points.
    {
    std::vector<cv::Point2f> rp, cp;
    for (int i = 0; i < inliersMask.rows; i++) {
        uchar im = inliersMask.at<uchar>(i), m = mask.at<uchar>(i);
        if (im == 0 || m == 0) {
            if (im == 0) mask.at<uchar>(i) = 0;
            continue;
        }
        rp.push_back(frame1Points[i]);
        cp.push_back(frame2Points[i]);
    }
    frame1Points = rp; frame2Points = cp;
    }
    // Calculate projection matrices.
    // TODO: use already estimated poses?
    cv::Mat firstProjection(3, 4, CV_32F, cv::Scalar(0));
    cameraMatrix.copyTo(firstProjection.rowRange(0, 3).colRange(0, 3));

    cv::Mat secondProjection(3, 4, CV_32F, cv::Scalar(0));
    rotation.copyTo(secondProjection.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(secondProjection.rowRange(0, 3).col(3));
    secondProjection = cameraMatrix * secondProjection;
    // Reconstruct points.
    cv::Mat reconstructedPointsM, homogeneousPoints;
    cv::triangulatePoints(
        firstProjection, secondProjection,
        frame1Points, frame2Points, homogeneousPoints
    );
    cv::convertPointsFromHomogeneous(homogeneousPoints.t(), reconstructedPointsM);
    auto reconstructedPoints = vectorFromMat(reconstructedPointsM);
    // Compose rotation and translation into pose matrix.
    cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
    rotation.copyTo(pose.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(pose.rowRange(0, 3).col(3));

    return {reconstructedPoints, pose, mask};
}

};
