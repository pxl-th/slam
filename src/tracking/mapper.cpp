#include"tracking/mapper.hpp"

namespace slam {

Mapper::Mapper(std::shared_ptr<Map> map) : map(map), acceptKeyframes(true) {}

bool Mapper::accepts() const { return acceptKeyframes; }

void Mapper::addKeyframe(std::shared_ptr<KeyFrame> keyframe) {
    keyframeQueue.push(keyframe);
}

void Mapper::_processKeyFrame(std::shared_ptr<KeyFrame> keyframe) {
    acceptKeyframes = false;

    currentKeyFrame = keyframeQueue.front();
    keyframeQueue.pop();

    _createConnections(currentKeyFrame);

    map->addKeyframe(keyframe);

    /**
     * TRIANGULATION:
     * for every keyframe connection, triangulate points between them,
     * then perform fusion for duplicates
     */
    /**
     * + create connections between keyframes before adding to the map
     * + add kf to map
     * - triangulate points
     */
}

void Mapper::_createConnections(
    std::shared_ptr<KeyFrame> targetKeyFrame, int threshold
) {
    targetKeyFrame->connections.clear();
    std::map<std::shared_ptr<KeyFrame>, int> counter;
    // Count number of mappoints that are shared
    // between each KeyFrame and this KeyFrame.
    for (const auto& [id, mappoint] : targetKeyFrame->getMapPoints()) {
        for (const auto& [keyframe, keypointId] : mappoint->getObservations()) {
            if (keyframe->id == targetKeyFrame->id) continue;
            counter[keyframe]++;
        }
    }
    // Create connections with KeyFrames that more than `threshold`
    // shared MapPoints.
    int maxCount = 0;
    for (const auto& [keyframe, count] : counter) {
        if (count > maxCount) maxCount = count;
        if (count < threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
    // If no KeyFrame's have been found --- add KeyFrame with maximum count.
    if (targetKeyFrame->connections.empty()) threshold = maxCount;
    for (const auto& [keyframe, count] : counter) {
        if (count != threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
}

};
