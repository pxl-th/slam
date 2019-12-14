#include<iostream>

#include "matcher.hpp"

namespace slam {

Matcher::Matcher(cv::Ptr<cv::BFMatcher> matcher)
    : matcher(matcher) {}

void Matcher::frameMatch(
    const Frame& frame1, const Frame& frame2,
    std::vector<cv::DMatch>& matches,
    float maximumDistance, float areaSize
) {
    matches.clear();
    bool inArea, checkArea = areaSize == -1;
    std::vector<cv::DMatch> descriptorMatches;
    matcher->match(frame1.descriptors, frame2.descriptors, descriptorMatches);

    cv::Point2f d;
    for (auto& m : descriptorMatches) {
        if (checkArea) {
            d = frame1.keypoints[m.queryIdx].pt - frame2.keypoints[m.trainIdx].pt;
            inArea = (abs(d.x) < areaSize && abs(d.y) < areaSize);
        } else inArea = true;

        if (inArea && m.distance < maximumDistance)
            matches.push_back(m);
    }
}

};
