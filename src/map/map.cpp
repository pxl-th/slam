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

};
