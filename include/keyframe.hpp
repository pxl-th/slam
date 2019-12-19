#ifndef KEYFRAME_H
#define KEYFRAME_H

#include<opencv2/core.hpp>

#include"frame.hpp"

namespace slam {

/**
 * Holder for Frames and camera position info.
 */
class KeyFrame {
private:
    int id;
    Frame frame;
    cv::Mat pose, cameraCenter;

public:
    KeyFrame(int id, const Frame& frame, const cv::Mat& pose);
    ~KeyFrame() = default;

    cv::Mat getPose() const;
    cv::Mat getCameraCenter() const;
    /**
     * Given pose matrix, update KeyFrame's pose and camera's center.
     *
     * @param pose New pose matrix.
     * Should contain both rotation and translation,
     * and have `3x4` shape.
     */
    void setPose(const cv::Mat& pose);
};

};

#endif
