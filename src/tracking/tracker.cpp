#pragma warning(push, 0)
#include<iostream>
#pragma warning(pop)

#include"tracking/tracker.hpp"
#include"tracking/optimizer.hpp"

namespace slam {

Tracker::Tracker(
    Calibration calibration, std::shared_ptr<Detector> detector, bool useMotion
) : detector(detector), useMotion(useMotion) {
    cameraMatrix = std::make_shared<cv::Mat>(calibration.cameraMatrix);
    distortions = std::make_shared<cv::Mat>(calibration.distortions);
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
        _trackFrame();

        // track motion frame
        // relocalisation
        // local mapper does triangulation!!!
        break;
    }
}

bool Tracker::_initialize() {
    auto initialFrame = initialKeyFrame->getFrame();
    auto currentFrame = currentKeyFrame->getFrame();

    initializer = Initializer(initialFrame);
    auto matches = matcher.frameMatch(initialFrame, currentFrame, 300, 50);
    std::cout << "Frame matches " << matches.size() << std::endl;
    if (matches.size() < 100) return false;

    auto [rotation, translation, mask, reconstructedPoints] = (
        initializer.initialize(currentFrame, matches)
    );
    std::cout << "Reconstructed points " << reconstructedPoints.size() << std::endl;

    map = initializer.initializeMap(
        currentFrame, rotation, translation, reconstructedPoints, matches, mask
    );
    return true;
}

void Tracker::_trackFrame() {
    auto matches = matcher.frameMatch(
        lastKeyFrame->getFrame(), currentKeyFrame->getFrame(), 300, 50
    );
    std::cout << "Tracking frame matches " << matches.size() << std::endl;

    // Add matched mappoints to current keyframe.
    auto lastMappoints = lastKeyFrame->getMapPoints();
    for (const auto& match : matches) {
        auto exist = lastMappoints.find(match.queryIdx);
        if (exist == lastMappoints.end()) continue;

        auto mappoint = exist->second;
        mappoint->addObservation(currentKeyFrame, match.trainIdx);
        currentKeyFrame->addMapPoint(match.trainIdx, mappoint);
    }
    std::cout
        << "Tracking mappoint matches "
        << currentKeyFrame->getMapPoints().size() << std::endl;

    currentKeyFrame->setPose(lastKeyFrame->getPose());
    optimizer::poseOptimization(currentKeyFrame);

    auto projectionMatches = matcher.projectionMatch(
        lastKeyFrame, currentKeyFrame, 200, 20
    );
    std::cout
        << "Tracking projection frame matches "
        << projectionMatches.size() << std::endl;
    for (const auto& match : projectionMatches) {
        auto exist = lastMappoints.find(match.queryIdx);
        if (exist == lastMappoints.end()) continue;

        auto mappoint = exist->second;
        mappoint->addObservation(currentKeyFrame, match.trainIdx);
        currentKeyFrame->addMapPoint(match.trainIdx, mappoint);
    }
    std::cout
        << "Total tracked points "
        << currentKeyFrame->getMapPoints().size() << std::endl;
}

void Tracker::_trackMotionFrame() {

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
