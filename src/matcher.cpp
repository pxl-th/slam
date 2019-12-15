#include<iostream>

#include "matcher.hpp"

namespace slam {

Matcher::Matcher(cv::Ptr<cv::BFMatcher> matcher) : matcher(matcher) {}

void Matcher::frameMatch(
    const Frame& frame1, const Frame& frame2,
    std::vector<cv::DMatch>& matches,
    float maximumDistance, float areaSize
) {
    matches.clear();
    bool inArea, checkArea = areaSize != -1;
    std::vector<cv::DMatch> descriptorMatches;
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
}

};
