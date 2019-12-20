#include"map/mappoint.hpp"

namespace slam {

MapPoint::MapPoint(const cv::Point3f& position, std::shared_ptr<KeyFrame> keyframe)
    : position(position), keyframe(keyframe) {}

cv::Point3f MapPoint::getWorldPos() const { return position; }

void MapPoint::setWorldPos(const cv::Point3f& newPos) { position = newPos; }

std::shared_ptr<KeyFrame> MapPoint::getReferenceKeyframe() const {
    return keyframe;
}

std::set<std::shared_ptr<KeyFrame>> MapPoint::getObservations() const {
    return observations;
}

void MapPoint::addObservation(std::shared_ptr<KeyFrame> keyframe) {
    observations.insert(keyframe);
}

};
