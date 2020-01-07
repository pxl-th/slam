#ifndef MAP_H
#define MAP_H

#pragma warning(push, 0)
#include<memory>
#include<vector>
#pragma warning(pop)

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

    void removeMappoint(std::shared_ptr<MapPoint>& mappoint);
};

};

#endif
