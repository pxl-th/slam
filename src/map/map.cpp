#pragma warning(push, 0)
#include<algorithm>
#pragma warning(pop)

#include"map/map.hpp"

namespace slam {

std::vector<std::shared_ptr<KeyFrame>> Map::getKeyframes() const {
    return keyframes;
}

void Map::addKeyframe(std::shared_ptr<KeyFrame> keyframe) {
    keyframes.push_back(keyframe);
}

std::vector<std::shared_ptr<MapPoint>> Map::getMappoints() const {
    return mapPoints;
}

void Map::addMappoint(std::shared_ptr<MapPoint> mappoint) {
    mapPoints.push_back(mappoint);
}

void Map::removeMappoint(std::shared_ptr<MapPoint>& mappoint) {
    auto p = std::find(mapPoints.begin(), mapPoints.end(), mappoint);
    if (p != mapPoints.end())
        mapPoints.erase(p);
}

};
