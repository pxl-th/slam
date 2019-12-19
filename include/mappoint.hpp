#ifndef MAPPOINT_H
#define MAPPOINT_H

#include<map>

#include"keyframe.hpp"

namespace slam {

class MapPoint {
private:
    int id;
    KeyFrame& keyframe;
    std::map<KeyFrame&, int> observations;

    cv::Mat position;
public:
    MapPoint(int id, KeyFrame& keyframe, const cv::Mat& position);
    ~MapPoint() = default;

    KeyFrame& getReferenceKeyframe() const;

    cv::Mat getWorlPos() const;
    void setWorldPos(const cv::Mat& newPos);

    std::map<KeyFrame&, int> getObservations() const;
    void addObservation(KeyFrame& keyframe, int id);
};

};

#endif
