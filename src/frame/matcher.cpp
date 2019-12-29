#pragma warning(push, 0)
#include<iostream>

#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include "frame/matcher.hpp"

namespace slam {

Matcher::Matcher(cv::Ptr<cv::BFMatcher> matcher) : matcher(matcher) {}

std::vector<cv::DMatch> Matcher::frameMatch(
    std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2,
    float maximumDistance, float areaSize
) {
    std::vector<cv::DMatch> matches, descriptorMatches;
    bool inArea, checkArea = areaSize != -1;
    cv::Point2f d;

    matcher->match(
        *frame1->descriptors, *frame2->descriptors, descriptorMatches
    );
    for (auto& m : descriptorMatches) {
        if (frame1->undistortedKeypoints[m.queryIdx].octave > 4)
            continue;

        if (checkArea) {
            d = (
                frame1->undistortedKeypoints[m.queryIdx].pt
                - frame2->undistortedKeypoints[m.trainIdx].pt
            );
            inArea = abs(d.x) < areaSize && abs(d.y) < areaSize;
        } else inArea = true;

        if (inArea && m.distance < maximumDistance)
            matches.push_back(m);
    }

    return matches;
}

std::vector<cv::DMatch> Matcher::projectionMatch(
    std::shared_ptr<KeyFrame> fromKeyFrame,
    std::shared_ptr<KeyFrame> toKeyFrame,
    float maximumDistance, float areaSize
) {
    auto projectedKeyPoints = _projectMapPoints(fromKeyFrame, toKeyFrame);
    std::vector<cv::DMatch> rawMatches, matches;
    matcher->match(
        *fromKeyFrame->getFrame()->descriptors,
        *toKeyFrame->getFrame()->descriptors,
        rawMatches
    );

    const bool checkArea = areaSize != -1;
    bool inArea; cv::Point2f diff, kp;
    const auto& toFrame = toKeyFrame->getFrame();
    for (const auto& m : rawMatches) {
        if (toFrame->undistortedKeypoints[m.queryIdx].octave > 4) continue;
        if (m.distance > maximumDistance) continue;
        // Check if matched `toKeyFrame` keypoint is in any of the
        // areas of `areaSize` around projected `fromKeyFrame` mappoints
        // to `toKeyFrame` image plane. Select only those matches.
        if (checkArea) {
            inArea = false;
            kp = toKeyFrame->getFrame()->undistortedKeypoints[m.trainIdx].pt;
            for (const auto& pp : projectedKeyPoints) {
                if (inArea) break;
                diff = kp - pp;
                inArea = abs(diff.x) < areaSize && abs(diff.y) < areaSize;
            }
            if (!inArea) continue;
        }
        matches.push_back(m);
    }
    return matches;
}

std::vector<cv::Point2f> Matcher::_projectMapPoints(
    std::shared_ptr<KeyFrame> fromKeyFrame,
    std::shared_ptr<KeyFrame> toKeyFrame
) {
    cv::Mat rotation, translation;
    auto currentPose = toKeyFrame->getPose();
    cv::Rodrigues(currentPose.rowRange(0, 3).colRange(0, 3), rotation);
    currentPose.rowRange(0, 3).col(3).copyTo(translation);

    std::vector<cv::Point3f> mappoints;
    for(const auto& [id, p] : fromKeyFrame->getMapPoints())
        mappoints.push_back(p->getWorldPos());

    std::vector<cv::Point2f> points;
    cv::projectPoints(
        mappoints, rotation, translation,
        *toKeyFrame->getFrame()->cameraMatrix,
        *toKeyFrame->getFrame()->distortions,
        points
    );

    return points;
}

};
