#pragma warning(push, 0)
#include<iostream>
#pragma warning(pop)

#include"tracking/tracker.hpp"

namespace slam {

Tracker::Tracker(Calibration calibration, std::shared_ptr<Detector> detector)
    : detector(detector) {
    cameraMatrix = std::make_shared<cv::Mat>(calibration.cameraMatrix);
    distortions = std::make_shared<cv::Mat>(calibration.distortions);
    state = NO_IMAGES;
}

void Tracker::track(std::shared_ptr<cv::Mat> image) {
    if (state == NO_IMAGES) { // TODO: correct timestamp
        initialFrame = packImage(image, 0);
        initializer = Initializer(initialFrame);
        state = UNINITIALIZED;
    } else if (state == UNINITIALIZED) {
        currentFrame = packImage(image, 0);
        if (initialize()) state = INITIALIZED;
    } else {
        std::cout << "hmmm" << std::endl;
    }
}

bool Tracker::initialize() {
    auto matches = matcher.frameMatch(initialFrame, currentFrame, 300, 50);
    std::cout << "Frame matches " << matches.size() << std::endl;
    if (matches.size() < 100) return false;

    auto [rotation, translation, mask, reconstructedPoints] = (
        initializer.initialize(currentFrame, matches)
    );
    std::cout << "Reconstructed points " << reconstructedPoints.size() << std::endl;
    std::cout << "Translation\n" << translation << std::endl;

    map = initializer.initializeMap(
        currentFrame, rotation, translation, reconstructedPoints, matches, mask
    );
    return true;
}

std::shared_ptr<Frame> Tracker::packImage(
    std::shared_ptr<cv::Mat> image, double timestamp
) {
    return std::make_shared<Frame>(
        image, timestamp, detector, cameraMatrix, distortions
    );
}

};
