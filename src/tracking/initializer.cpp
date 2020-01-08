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
    reference->setPose(cv::Mat::eye(4, 4, CV_32F)); // Wrong pose!!!! for reprojection
    current->setPose(pose);
    map->addKeyframe(reference);
    map->addKeyframe(current);

    // Add observations to mappoints and add mappoints to map.
    for (size_t i = 0, j = 0; i < matches.size(); i++) {
        if (outliersMask.at<uchar>(static_cast<int>(i)) == 0) continue;
        auto point = reconstructedPoints[j++];
        if (isOutlier(point, reference, current, matches[i]))
            continue;

        auto mappoint = std::make_shared<MapPoint>(point, current);
        mappoint->addObservation(reference, matches[i].queryIdx);
        mappoint->addObservation(current, matches[i].trainIdx);

        reference->addMapPoint(matches[i].queryIdx, mappoint);
        current->addMapPoint(matches[i].trainIdx, mappoint);

        map->addMappoint(mappoint);
    }

    float inverseMedianDepth = 1.0f / reference->medianDepth();
    std::cout << inverseMedianDepth << std::endl;
    // TODO: assert that depth is positive, why?

    // Scale translation by inverse median depth.
    auto currentPose = current->getPose();
    currentPose.col(3).rowRange(0, 3) = (
        currentPose.col(3).rowRange(0, 3) * inverseMedianDepth
    );
    current->setPose(currentPose);
    // Scale mappoints by inverse median depth.
    for (auto& [id, p] : reference->getMapPoints())
        p->setWorldPos(p->getWorldPos() * inverseMedianDepth);

    optimizer::globalBundleAdjustment(map, 20);
    return map;
}

};
