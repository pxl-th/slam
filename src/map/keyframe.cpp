#include"map/keyframe.hpp"

namespace slam {

unsigned long long KeyFrame::globalID = 0;

KeyFrame::KeyFrame(const Frame& frame, const cv::Mat& pose)
    : id(globalID++), frame(frame) {
    setPose(pose);
}

void KeyFrame::setPose(const cv::Mat& pose) {
    pose.copyTo(this->pose);
    cameraCenter = (
        (-pose.rowRange(0, 3).colRange(0, 3).t())
        * pose.col(3).rowRange(0, 3)
    );
}

cv::Mat KeyFrame::getPose() const { return pose.clone(); }

cv::Mat KeyFrame::getCameraCenter() const { return cameraCenter.clone(); }

const Frame& KeyFrame::getFrame() const { return frame; }

void KeyFrame::addMapPoint(std::shared_ptr<MapPoint> mapPoint) {
    mapPoints.insert(mapPoint);
}

std::set<std::shared_ptr<MapPoint>> KeyFrame::getMapPoints() const {
    return mapPoints;
}

};
