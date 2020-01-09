#pragma warning(push, 0)
#include<iostream>

#include<opencv2/core/cvdef.h>
#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include"converter.hpp"
#include"map/mappoint.hpp"

namespace slam {

MapPoint::MapPoint(const cv::Point3f& position, std::shared_ptr<KeyFrame> keyframe)
    : position(position), keyframe(keyframe) {}

cv::Point3f MapPoint::getWorldPos() const { return position; }

void MapPoint::setWorldPos(const cv::Point3f& newPos) { position = newPos; }

std::shared_ptr<KeyFrame> MapPoint::getReferenceKeyframe() const {
    return keyframe;
}

std::map<std::shared_ptr<KeyFrame>, int> MapPoint::getObservations() const {
    return observations;
}

void MapPoint::addObservation(std::shared_ptr<KeyFrame> keyframeO, int id) {
    observations[keyframeO] = id;
}

void MapPoint::removeObservation(std::shared_ptr<KeyFrame> keyframeO) {
    auto mp = observations.find(keyframeO);
    if (mp != observations.end())
        observations.erase(mp);
}

double parallax(cv::Mat point, cv::Mat camera1, cv::Mat camera2, bool radians) {
    cv::Mat normal1 = point - camera1, normal2 = point - camera2;

    double p = normal1.dot(normal2) / (cv::norm(normal1) * cv::norm(normal2));
    if (radians) return p;
    p  = std::acos(p) * 180.0 / CV_PI;
    return p;
}

double parallax(
    cv::Point3f point, cv::Mat camera1, cv::Mat camera2, bool radians
) { return parallax(matFromPoint3f(point).t(), camera1, camera2, radians); }

bool isOutlier(
    const cv::Point3f& point,
    const std::shared_ptr<KeyFrame>& queryKeyframe,
    const std::shared_ptr<KeyFrame>& trainKeyframe,
    const cv::DMatch& match
) {
    auto pointMat = matFromPoint3f(point).t();
    // Check that parallax for a point lies in (0, 1) range.
    // Otherwise point considered an outlier and discarded.
    auto pointParallax = parallax(
        pointMat,
        queryKeyframe->getCameraCenter(),
        trainKeyframe->getCameraCenter()
    );
    if (pointParallax < 0.0f || pointParallax > 0.999f) return true;
    // Check that point does not lie near one of keyframe's center.
    if (
        std::abs(cv::norm(pointMat - queryKeyframe->getCameraCenter())) < 1E-6
        || std::abs(cv::norm(pointMat - trainKeyframe->getCameraCenter())) < 1E-6
    ) return true;
    // Check reprojection error.
    if (
        projectionError(queryKeyframe, point, match.queryIdx) > 1 ||
        projectionError(trainKeyframe, point, match.trainIdx) > 1
    ) return true;
    return false;
}

double projectionError(
    const std::shared_ptr<KeyFrame>& keyframe,
    const cv::Point3f& point,
    int keypointId
) {
    cv::Mat rotation;
    cv::Rodrigues(keyframe->getPose().rowRange(0, 3).colRange(0, 3), rotation);
    cv::Mat translation = keyframe->getPose().rowRange(0, 3).col(3);

    std::vector<cv::Point2f> projection;
    cv::projectPoints(
        std::vector<cv::Point3f>{point}, rotation, translation,
        *keyframe->getFrame()->cameraMatrix, *keyframe->getFrame()->distortions,
        projection
    );

    auto target = keyframe->getFrame()->undistortedKeypoints[keypointId].pt;
    auto p = cv::norm(target - projection[0]);
    return p;
}

};
