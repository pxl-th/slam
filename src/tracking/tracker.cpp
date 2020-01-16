#pragma warning(push, 0)
#include<iostream>
#pragma warning(pop)

#include"converter.hpp"
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
    std::cout << "[tracking] Packing image" << std::endl;
    currentKeyFrame = _packImage(image);
    std::cout << "[tracking] State " << state << std::endl;

    switch (state) {
    case NO_IMAGES:
        lastKeyFrame = currentKeyFrame;
        state = UNINITIALIZED;
        break;
    case UNINITIALIZED:
        mapper.addKeyframe(lastKeyFrame);
        mapper.addKeyframe(currentKeyFrame);
        if (mapper.initialize()) state = INITIALIZED;
        lastKeyFrame = currentKeyFrame;
        break;
    case INITIALIZED:
        const bool motionTracking = (
            useMotion
            && !velocity.empty()
            && (mapper.map->getKeyframes().size() >= 4)
        );

        std::cout
            << "[tracking] last keyframe "
            << lastKeyFrame->mappointsNumber() << std::endl;
        bool successfulTracking = false;
        if (motionTracking) successfulTracking = _trackMotionFrame();
        if (!successfulTracking) successfulTracking = _trackFrame();
        if (useMotion) _updateMotion(successfulTracking);
        std::cout
            << "[tracking] Successful tracking "
            << successfulTracking << std::endl;

        if (!successfulTracking) {
            state = LOST;
            return;
        }

        if (currentKeyFrame->mappointsNumber() < 50) {
            mapper.addKeyframe(currentKeyFrame);
            mapper.process();
        } else {
            for (const auto [i, mappoint] : currentKeyFrame->mappoints)
                mappoint->removeObservation(currentKeyFrame);
        }

        lastKeyFrame = currentKeyFrame;
        break;
    }
}

bool Tracker::_trackFrame() {
    std::cout << "[tracking] Frame" << std::endl;
    // Add matched mappoints to current keyframe.
    auto matches = matcher.mappointsFrameMatch(lastKeyFrame, currentKeyFrame, 300, 50);
    _addMatches(currentKeyFrame, lastKeyFrame, matches);
    if (currentKeyFrame->mappointsNumber() < 30) {
        matches = matcher.mappointsFrameMatch(lastKeyFrame, currentKeyFrame, 300, -1, -1);
        _addMatches(currentKeyFrame, lastKeyFrame, matches);
    }
    std::cout
        << "[tracking] matches "
        << currentKeyFrame->mappointsNumber() << std::endl;
    currentKeyFrame->setPose(lastKeyFrame->getPose());
    // TODO: do not optimize if no mappoints
    optimizer::poseOptimization(currentKeyFrame);

    matches = matcher.projectionMatch(lastKeyFrame, currentKeyFrame, 300, 50);
    _addMatches(currentKeyFrame, lastKeyFrame, matches);
    std::cout
        << "[tracking] matches "
        << currentKeyFrame->mappointsNumber() << std::endl;
    optimizer::poseOptimization(currentKeyFrame);
    return currentKeyFrame->mappointsNumber() >= 5;
}

bool Tracker::_trackMotionFrame() {
    std::cout << "[tracking] Motion" << std::endl;
    currentKeyFrame->setPose(velocity * lastKeyFrame->getPose());

    auto matches = matcher.projectionMatch(lastKeyFrame, currentKeyFrame, 300, 50);
    _addMatches(currentKeyFrame, lastKeyFrame, matches);
    if (currentKeyFrame->mappointsNumber() < 30) {
        matches = matcher.projectionMatch(lastKeyFrame, currentKeyFrame, 300, -1, -1);
        _addMatches(currentKeyFrame, lastKeyFrame, matches);
    }
    optimizer::poseOptimization(currentKeyFrame);
    return currentKeyFrame->mappointsNumber() >= 5;
}

void Tracker::_addMatches(
    std::shared_ptr<KeyFrame>& keyframe,
    const std::shared_ptr<KeyFrame>& lastKeyframe,
    const std::vector<cv::DMatch>& matches,
    bool checkMatches
) {
    for (const auto& match : matches) {
        std::shared_ptr<MapPoint> mappoint;
        if (checkMatches) {
            auto exist = lastKeyFrame->mappoints.find(match.queryIdx);
            if (exist == lastKeyFrame->mappoints.end()) continue;
            mappoint = exist->second;
        } else
            mappoint = lastKeyframe->mappoints[match.queryIdx];
        mappoint->addObservation(keyframe, match.trainIdx);
        keyframe->addMapPoint(match.trainIdx, mappoint);
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

std::shared_ptr<KeyFrame> Tracker::_packImage(std::shared_ptr<cv::Mat> image) {
    return std::make_shared<KeyFrame>(
        std::make_shared<Frame>(image, detector, cameraMatrix, distortions),
        cv::Mat::eye(4, 4, CV_32F)
    );
}

};
