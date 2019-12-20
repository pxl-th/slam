#ifndef MAPPOINT_H
#define MAPPOINT_H

#include<set>

#include"keyframe.hpp"

namespace slam {

class KeyFrame;

class MapPoint {
private:
    std::shared_ptr<KeyFrame> keyframe;
    std::set<std::shared_ptr<KeyFrame>> observations;

    cv::Point3f position;
public:
    MapPoint(const cv::Point3f& position, std::shared_ptr<KeyFrame> keyframe);

    std::shared_ptr<KeyFrame> getReferenceKeyframe() const;

    cv::Point3f getWorldPos() const;
    void setWorldPos(const cv::Point3f& newPos);

    std::set<std::shared_ptr<KeyFrame>> getObservations() const;
    void addObservation(std::shared_ptr<KeyFrame> keyframe);
};

};

#endif
