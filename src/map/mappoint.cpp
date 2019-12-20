#include"map/mappoint.hpp"

namespace slam {

MapPoint::MapPoint(int id, KeyFrame& keyframe, const cv::Mat& position)
    : id(id), keyframe(keyframe) {
    position.copyTo(this->position);
}

cv::Mat MapPoint::getWorlPos() const { return position.clone(); }

void MapPoint::setWorldPos(const cv::Mat& newPos) { newPos.copyTo(position); }

KeyFrame& MapPoint::getReferenceKeyframe() const { return keyframe; }

std::map<KeyFrame*, int> MapPoint::getObservations() const {
    return observations;
}

void MapPoint::addObservation(KeyFrame& keyframe, int id) {
    observations[&keyframe] = id;
}

};
