#pragma warning(push, 0)
#include<iostream>
#include<opencv2/core/cvdef.h>
#pragma warning(pop)

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

double MapPoint::parallax(cv::Mat point, cv::Mat camera1, cv::Mat camera2) {
    cv::Mat normal1 = point - camera1, normal2 = point - camera2;

    double p = normal1.dot(normal2) / (cv::norm(normal1) * cv::norm(normal2));
    p  = std::acos(p) * 180.0 / CV_PI;
    return p;
}

};
