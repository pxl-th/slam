#pragma warning(push, 0)
#include<algorithm>
#pragma warning(pop)

#include"converter.hpp"
#include"map/keyframe.hpp"

namespace slam {

unsigned long long KeyFrame::globalID = 0;

KeyFrame::KeyFrame(std::shared_ptr<Frame> frame, const cv::Mat& pose)
    : id(globalID++), frame(frame) {
    setPose(pose);
}

void KeyFrame::setPose(const cv::Mat& newPose) {
    newPose.copyTo(pose);
    cameraCenter = (
        (-pose.rowRange(0, 3).colRange(0, 3).t())
        * pose.col(3).rowRange(0, 3)
    );
}

cv::Mat KeyFrame::getPose() const { return pose.clone(); }

cv::Mat KeyFrame::getCameraCenter() const { return cameraCenter.clone(); }

std::shared_ptr<Frame> KeyFrame::getFrame() const { return frame; }

void KeyFrame::addMapPoint(int keypointId, std::shared_ptr<MapPoint> mapPoint) {
    mappoints[keypointId] = mapPoint;
}

void KeyFrame::removeMapPoint(int keypointId) {
    auto mp = mappoints.find(keypointId);
    if (mp != mappoints.end())
        mappoints.erase(mp);
}

std::map<int, std::shared_ptr<MapPoint>> KeyFrame::getMapPoints() const {
    return mappoints;
}

size_t KeyFrame::mappointsNumber() const {
    return mappoints.size();
}

float KeyFrame::medianDepth() const {
    std::vector<float> depths;
    const cv::Mat depthTransformation = pose.row(2).colRange(0, 3).t();
    const float depthTranslation = pose.at<float>(2, 3);

    for (const auto& [i, p] : mappoints) {
        depths.push_back(static_cast<float>(
            depthTransformation.dot(cv::Mat(p->getWorldPos(), false))
        ) + depthTranslation);
    }
    std::sort(depths.begin(), depths.end());
    return depths[depths.size() / 2];
}

float KeyFrame::medianDepth(std::vector<std::shared_ptr<MapPoint>> points) const {
    std::vector<float> depths;
    const cv::Mat depthTransformation = pose.row(2).colRange(0, 3).t();
    const float depthTranslation = pose.at<float>(2, 3);

    for (const auto& p : points) {
        depths.push_back(static_cast<float>(
            depthTransformation.dot(cv::Mat(p->getWorldPos(), false))
        ) + depthTranslation);
    }
    std::sort(depths.begin(), depths.end());
    return depths[depths.size() / 2];
}

};
