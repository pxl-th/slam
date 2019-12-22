#ifndef KEYFRAME_H
#define KEYFRAME_H

#pragma warning(push, 0)
#include<set>

#include<opencv2/core.hpp>
#pragma warning(pop)

#include"frame/frame.hpp"
#include"mappoint.hpp"

namespace slam {

class MapPoint;

/**
 * Holder for Frames and camera position info.
 */
class KeyFrame {
private:
    Frame frame;
    cv::Mat pose, cameraCenter;

    /**
     * Set of MapPoints that are visible from this KeyFrame.
     */
    std::set<std::shared_ptr<MapPoint>> mapPoints;
public:
    static unsigned long long globalID;
    unsigned long long id;
public:
    KeyFrame(const Frame& frame, const cv::Mat& pose);

    cv::Mat getPose() const;
    cv::Mat getCameraCenter() const;
    const Frame& getFrame() const;

    /**
     * Given pose matrix, update KeyFrame's pose and camera's center.
     *
     * @param pose New pose matrix.
     * Should contain both rotation and translation,
     * and have `4x4` shape.
     */
    void setPose(const cv::Mat& pose);

    void addMapPoint(std::shared_ptr<MapPoint> mapPoint);

    std::set<std::shared_ptr<MapPoint>> getMapPoints() const;
};

};

#endif
