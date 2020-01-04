#pragma warning(push, 0)
#include<iostream>

#include<opencv2/core.hpp>
#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include"converter.hpp"
#include"tracking/initializer.hpp"
#include"tracking/mapper.hpp"
#include"tracking/optimizer.hpp"

namespace slam {

Initializer::Initializer(std::shared_ptr<KeyFrame> reference)
    : reference(reference) {}

std::shared_ptr<Map> Initializer::initializeMap(
    const std::shared_ptr<KeyFrame> current, const cv::Mat& pose,
    const std::vector<cv::Point3f>& reconstructedPoints,
    const std::vector<cv::DMatch>& matches, const cv::Mat& outliersMask
) {
    auto map = std::make_shared<Map>();
    current->setPose(pose);
    // current -- query, reference -- train
    map->addKeyframe(reference);
    map->addKeyframe(current);

    // Add observations to mappoints and add mappoints to map.
    for (size_t i = 0, j = 0; i < matches.size(); i++) {
        if (outliersMask.at<uchar>(static_cast<int>(i)) == 0) continue;

        auto mappoint = std::make_shared<MapPoint>(
            reconstructedPoints[j++], current
        );

        mappoint->addObservation(current, matches[i].queryIdx);
        mappoint->addObservation(reference, matches[i].trainIdx);

        current->addMapPoint(matches[i].queryIdx, mappoint);
        reference->addMapPoint(matches[i].trainIdx, mappoint);

        map->addMappoint(mappoint);
    }

    optimizer::globalBundleAdjustment(map, 20);

    float inverseMedianDepth = 1.0f / reference->medianDepth();
    // TODO: assert that depth is positive

    // Scale translation by inverse median depth.
    auto currentPose = current->getPose();
    currentPose.col(3).rowRange(0, 3) = (
        currentPose.col(3).rowRange(0, 3) * inverseMedianDepth
    );
    current->setPose(currentPose);
    // Scale mappoints by inverse median depth.
    for (auto& [id, p] : reference->getMapPoints())
        p->setWorldPos(p->getWorldPos() * inverseMedianDepth);

    return map;
}

};
