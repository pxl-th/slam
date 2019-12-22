#pragma warning(push, 0)
#include<iostream>
#pragma warning(pop)

#include "frame/matcher.hpp"

namespace slam {

Matcher::Matcher(cv::Ptr<cv::BFMatcher> matcher) : matcher(matcher) {}

std::vector<cv::DMatch> Matcher::frameMatch(
    Frame& frame1, Frame& frame2,
    float maximumDistance, float areaSize
) {
    std::vector<cv::DMatch> matches, descriptorMatches;
    bool inArea, checkArea = areaSize != -1;
    cv::Point2f d;

    matcher->match(frame1.descriptors, frame2.descriptors, descriptorMatches);
    for (auto& m : descriptorMatches) {
        if (frame1.undistortedKeypoints[m.queryIdx].octave > 4)
            continue;

        if (checkArea) {
            d = (
                frame1.undistortedKeypoints[m.queryIdx].pt
                - frame2.undistortedKeypoints[m.trainIdx].pt
            );
            inArea = (abs(d.x) < areaSize && abs(d.y) < areaSize);
        } else inArea = true;

        if (inArea && m.distance < maximumDistance)
            matches.push_back(m);
    }

    return matches;
}

};
