#include"map/keyframe.hpp"

namespace slam {

KeyFrame::KeyFrame(int id, const Frame& frame, const cv::Mat& pose)
    : id(id), frame(frame) {
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

};
