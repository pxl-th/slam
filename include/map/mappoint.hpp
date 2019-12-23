#ifndef MAPPOINT_H
#define MAPPOINT_H

#pragma warning(push, 0)
#include<map>
#pragma warning(pop)

#include"keyframe.hpp"

namespace slam {

class KeyFrame;

class MapPoint {
private:
    cv::Point3f position;
    std::shared_ptr<KeyFrame> keyframe;
    std::map<std::shared_ptr<KeyFrame>, int> observations;
public:
    MapPoint(const cv::Point3f& position, std::shared_ptr<KeyFrame> keyframe);

    std::shared_ptr<KeyFrame> getReferenceKeyframe() const;

    cv::Point3f getWorldPos() const;
    void setWorldPos(const cv::Point3f& newPos);

    std::map<std::shared_ptr<KeyFrame>, int> getObservations() const;
    void addObservation(std::shared_ptr<KeyFrame> keyframe, int id);
};

};

#endif
