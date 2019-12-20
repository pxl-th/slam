#ifndef MAP_H
#define MAP_H

#include<memory>
#include<vector>

#include"mappoint.hpp"
#include"keyframe.hpp"

namespace slam {

class Map {
private:
    std::vector<std::shared_ptr<KeyFrame>> keyframes;
    std::vector<std::shared_ptr<MapPoint>> mapPoints;
public:
    std::vector<std::shared_ptr<KeyFrame>> getKeyframes() const;

    void addKeyframe(std::shared_ptr<KeyFrame> keyframe);

    std::vector<std::shared_ptr<MapPoint>> getMappoints() const;

    void addMappoint(std::shared_ptr<MapPoint> mappoint);
};

};

#endif
