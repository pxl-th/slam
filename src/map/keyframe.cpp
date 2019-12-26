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

void KeyFrame::addMapPoint(std::shared_ptr<MapPoint> mapPoint) {
    mapPoints.push_back(mapPoint);
}

std::vector<std::shared_ptr<MapPoint>> KeyFrame::getMapPoints() const {
    return mapPoints;
}

float KeyFrame::medianDepth() const {
    std::vector<float> depths;
    depths.resize(mapPoints.size());

    const cv::Mat depthTransformation = pose.row(2).colRange(0, 3).t();
    const float depthTranslation = pose.at<float>(2, 3);

    for (size_t i = 0; i < mapPoints.size(); i++) {
        depths[i] = (
            static_cast<float>(depthTransformation.dot(
                cv::Mat(mapPoints[i]->getWorldPos(), false)
            )) + depthTranslation
        );
    }
    std::sort(depths.begin(), depths.end());
    return depths[depths.size() / 2];
}

};
