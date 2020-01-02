#pragma warning(push, 0)
#include<iostream>
#pragma warning(pop)

#include"tracking/mapper.hpp"
#include"tracking/tracker.hpp"
#include"tracking/optimizer.hpp"

namespace slam {

Tracker::Tracker(
    Calibration calibration, std::shared_ptr<Detector> detector, bool useMotion
) : detector(detector), useMotion(useMotion) {
    cameraMatrix = std::make_shared<cv::Mat>(calibration.cameraMatrix);
    distortions = std::make_shared<cv::Mat>(calibration.distortions);
    velocity = cv::Mat();
    state = NO_IMAGES;
}

void Tracker::track(std::shared_ptr<cv::Mat> image) {
    // TODO: correct timestamp
    currentKeyFrame = _packImage(image, 0);

    switch (state) {
    case NO_IMAGES:
        initialKeyFrame = currentKeyFrame;
        state = UNINITIALIZED;
        break;
    case UNINITIALIZED:
        if (_initialize()) state = INITIALIZED;
        lastKeyFrame = map->getKeyframes()[1];
        break;
    case INITIALIZED:
        // TODO: check for relocalisation before tracking
        const bool motionTracking = (
            useMotion
            && !velocity.empty()
            && (map->getKeyframes().size() >= 4)
        );

        bool successfulTracking = false;
        if (motionTracking) successfulTracking = _trackMotionFrame();
        if (!successfulTracking) successfulTracking = _trackFrame();
        std::cout
            << "[tracking] Successful tracking "
            << successfulTracking << std::endl;
        if (useMotion) _updateMotion(successfulTracking);

        std::cout
            << "[tracking] Before addition "
            << currentKeyFrame->getMapPoints().size() << std::endl;
        mapper.addKeyframe(currentKeyFrame);
        std::cout
            << "[tracking] After addition "
            << currentKeyFrame->getMapPoints().size() << std::endl;
        lastKeyFrame = currentKeyFrame;
        break;
    }
}

bool Tracker::_initialize() {
    auto initialFrame = initialKeyFrame->getFrame();
    auto currentFrame = currentKeyFrame->getFrame();

    initializer = Initializer(initialKeyFrame);
    auto matches = matcher.frameMatch(initialFrame, currentFrame, 300, 50);
    std::cout << "Frame matches " << matches.size() << std::endl;
    if (matches.size() < 100) return false;

    auto [reconstructedPoints, pose, mask] = std::get<0>(Mapper::triangulatePoints(
        initialKeyFrame, currentKeyFrame, matches, true
    ));
    std::cout << "Reconstructed points " << reconstructedPoints.size() << std::endl;
    map = initializer.initializeMap(
        currentKeyFrame, pose, reconstructedPoints, matches, mask
    );
    mapper.map = map;
    return true;
}

bool Tracker::_trackFrame() {
    std::cout << "Tracking..." << std::endl;
    const auto& lastMappoints = lastKeyFrame->getMapPoints();
    std::cout << "Mappoints number " << lastMappoints.size() << std::endl;

    // Add matched mappoints to current keyframe.
    auto matches = matcher.frameMatch(
        lastKeyFrame->getFrame(), currentKeyFrame->getFrame(), 300, 50
    );
    std::cout << "Frame matches " << matches.size() << std::endl;
    _addMatches(currentKeyFrame, lastMappoints, matches);
    std::cout << "Current matches " << currentKeyFrame->getMapPoints().size() << std::endl;
    currentKeyFrame->setPose(lastKeyFrame->getPose());
    optimizer::poseOptimization(currentKeyFrame);

    auto projectionMatches = matcher.projectionMatch(
        lastKeyFrame, currentKeyFrame, 200, 20
    );
    _addMatches(currentKeyFrame, lastMappoints, matches);
    optimizer::poseOptimization(currentKeyFrame);

    std::cout
        << "Total tracked points "
        << currentKeyFrame->getMapPoints().size() << std::endl;
    return currentKeyFrame->getMapPoints().size() >= 10;
}

bool Tracker::_trackMotionFrame() {
    std::cout << "Motion tracking..." << std::endl;
    currentKeyFrame->setPose(velocity * lastKeyFrame->getPose());

    auto projectionMatches = matcher.projectionMatch(
        lastKeyFrame, currentKeyFrame, 200, 20
    );
    _addMatches(currentKeyFrame, lastKeyFrame->getMapPoints(), projectionMatches);
    optimizer::poseOptimization(currentKeyFrame);

    std::cout
        << "Total motion tracked points "
        << currentKeyFrame->getMapPoints().size() << std::endl;
    return currentKeyFrame->getMapPoints().size() >= 10;
}

void Tracker::_addMatches(
    std::shared_ptr<KeyFrame> keyframe,
    const std::map<int, std::shared_ptr<MapPoint>>& lastMappoints,
    const std::vector<cv::DMatch>& matches
) {
    for (const auto& match : matches) {
        auto exist = lastMappoints.find(match.queryIdx);
        if (exist == lastMappoints.end()) continue;

        auto& mappoint = exist->second;
        mappoint->addObservation(currentKeyFrame, match.trainIdx);
        currentKeyFrame->addMapPoint(match.trainIdx, mappoint);
    }
}

void Tracker::_updateMotion(bool successfulTracking) {
    if (!successfulTracking) {
        velocity = cv::Mat();
        return;
    }

    cv::Mat lastMotion = cv::Mat::eye(4, 4, CV_32F);
    auto lastPose = lastKeyFrame->getPose();
    cv::Mat lastRotation = lastPose.rowRange(0, 3).colRange(0, 3).t();
    cv::Mat lastTranslation = -lastRotation * lastPose.rowRange(0, 3).col(3);

    lastRotation.copyTo(lastMotion.rowRange(0, 3).colRange(0, 3));
    lastTranslation.copyTo(lastMotion.rowRange(0, 3).col(3));

    velocity = currentKeyFrame->getPose() * lastMotion;
}

std::shared_ptr<KeyFrame>
Tracker::_packImage(std::shared_ptr<cv::Mat> image, double timestamp) {
    auto frame = std::make_shared<Frame>(
        image, timestamp, detector, cameraMatrix, distortions
    );
    return std::make_shared<KeyFrame>(frame, cv::Mat::eye(4, 4, CV_32F));
}

};
