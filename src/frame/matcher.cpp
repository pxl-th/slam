#pragma warning(push, 0)
#include<algorithm>
#include<iostream>
#include<unordered_set>

#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include "frame/matcher.hpp"

namespace slam {

Matcher::Matcher(cv::Ptr<cv::BFMatcher> matcher) : matcher(matcher) {}

std::vector<cv::DMatch> Matcher::frameMatch(
    const std::shared_ptr<KeyFrame>& keyframe1,
    const std::shared_ptr<KeyFrame>& keyframe2,
    const std::vector<int>& ids,
    float maximumDistance, float areaSize, int maxLevel
) const {
    const auto& frame1 = keyframe1->getFrame(), frame2 = keyframe2->getFrame();
    // Select descriptors for existing mappoints in `keyframe1`.
    cv::Mat descriptors1;
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<int> idMapping;
    if (!ids.empty()) {
        descriptors1 = cv::Mat(static_cast<int>(ids.size()), 32, CV_8U);
        int rowId = 0;
        for (const int id : ids) {
            idMapping.push_back(id);
            keypoints1.push_back(frame1->undistortedKeypoints[id]);
            const auto& row = frame1->descriptors->row(id);
            for (int i = 0; i < 32; i++)
                descriptors1.at<uchar>(rowId, i) = row.at<uchar>(0, i);
            ++rowId;
        }
    } else {
        descriptors1 = *frame1->descriptors;
        keypoints1 = frame1->undistortedKeypoints;
    }
    // Find matches.
    std::vector<std::vector<cv::DMatch>> descriptorMatches;
    matcher->radiusMatch(
        descriptors1, *frame2->descriptors, descriptorMatches, maximumDistance
    );
    // Filter out matches by pixel-distance and keypoint's octave.
    std::vector<cv::DMatch> finalMatches = _filterMatches(
        keypoints1, frame2->undistortedKeypoints, descriptorMatches,
        areaSize, maxLevel
    );
    if (!ids.empty()) // Map query matches to `keyframe1` original KeyPoints.
        for (auto& match : finalMatches)
            match.queryIdx = idMapping[match.queryIdx];
    return finalMatches;
}

std::vector<cv::DMatch> Matcher::mappointsFrameMatch(
    const std::shared_ptr<KeyFrame>& keyframe1,
    const std::shared_ptr<KeyFrame>& keyframe2,
    float maximumDistance, float areaSize, int maxLevel
) const {
    std::vector<int> ids;
    for (const auto& [id, mappoint] : keyframe1->mappoints)
        ids.push_back(id);
    return frameMatch(keyframe1, keyframe2, ids, maximumDistance, areaSize, maxLevel);
}

std::vector<cv::DMatch> Matcher::inverseMappointsFrameMatch(
    const std::shared_ptr<KeyFrame>& keyframe1,
    const std::shared_ptr<KeyFrame>& keyframe2,
    float maximumDistance, float areaSize, int maxLevel
) const {
    if (keyframe1->mappoints.empty())
        return frameMatch(keyframe1, keyframe2, {}, maximumDistance, areaSize, maxLevel);

    std::vector<int> ids, mappointIds, matchIds;
    const auto& frame1 = keyframe1->getFrame();
    for (int i = 0; i < static_cast<int>(frame1->undistortedKeypoints.size()); i++)
        ids.push_back(i);
    for (const auto& [id, mappoint] : keyframe1->mappoints)
        mappointIds.push_back(id);

    std::set_difference(
        ids.begin(), ids.end(), mappointIds.begin(), mappointIds.end(),
        std::inserter(matchIds, matchIds.begin())
    );
    ids = matchIds;
    return frameMatch(keyframe1, keyframe2, ids, maximumDistance, areaSize, maxLevel);
}

std::vector<cv::DMatch> Matcher::projectionMatch(
    const std::shared_ptr<KeyFrame>& fromKeyFrame,
    const std::shared_ptr<KeyFrame>& toKeyFrame,
    float maximumDistance, float areaSize, int maxLevel
) const {
    const auto& frame1 = fromKeyFrame->getFrame(), frame2 = toKeyFrame->getFrame();
    // Select descriptors for existing mappoints in `keyframe1`.
    cv::Mat descriptors1(static_cast<int>(fromKeyFrame->mappoints.size()), 32, CV_8U);
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<int> idMapping;
    int rowId = 0;
    for (const auto& [id, mappoint] : fromKeyFrame->mappoints) {
        idMapping.push_back(id);
        keypoints1.push_back(frame1->undistortedKeypoints[id]);
        const auto& row = frame1->descriptors->row(id);
        for (int i = 0; i < 32; i++)
            descriptors1.at<uchar>(rowId, i) = row.at<uchar>(0, i);
        ++rowId;
    }
    // Find matches.
    std::vector<std::vector<cv::DMatch>> descriptorMatches;
    matcher->radiusMatch(
        descriptors1, *frame2->descriptors, descriptorMatches, maximumDistance
    );
    bool checkArea = areaSize != -1;
    std::vector<cv::Point2f> projectedKeyPoints;
    if (checkArea) {// Project `fromKeyFrame`'s mappoint onto `toKeyFrame`'s image plane.
        projectedKeyPoints = _projectMapPoints(fromKeyFrame, toKeyFrame);
        for (size_t i = 0; i < projectedKeyPoints.size(); i++)
            keypoints1[i].pt = projectedKeyPoints[i];
    }
    // Filter out matches by pixel-distance and keypoint's octave.
    std::vector<cv::DMatch> finalMatches = _filterMatches(
        keypoints1, frame2->undistortedKeypoints, descriptorMatches,
        areaSize, maxLevel
    );
    // Map query matches to `keyframe1` original KeyPoints.
    for (auto& match : finalMatches) match.queryIdx = idMapping[match.queryIdx];
    return finalMatches;
}

std::vector<cv::DMatch> Matcher::_filterMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<std::vector<cv::DMatch>>& rawMatches,
    float areaSize, int maxLevel
) {
    std::vector<cv::DMatch> finalMatches;
    cv::Point2f distance;
    bool noAreaCheck = areaSize == -1;
    for (const auto& matches : rawMatches) {
        if (matches.empty()) continue;
        const auto& keypoint1 = keypoints1[matches[0].queryIdx];
        for (const auto& match : matches) {
            if (
                maxLevel != -1
                && keypoint1.octave > maxLevel
                && keypoints2[match.trainIdx].octave > maxLevel
            ) continue;
            if (noAreaCheck) {
                finalMatches.push_back(match);
                break;
            }
            distance = keypoint1.pt - keypoints2[match.trainIdx].pt;
            if (abs(distance.x) < areaSize && abs(distance.y) < areaSize) {
                finalMatches.push_back(match);
                break;
            }
        }
    }
    return finalMatches;
}

std::vector<cv::Point2f> Matcher::_projectMapPoints(
    const std::shared_ptr<KeyFrame>& fromKeyFrame,
    const std::shared_ptr<KeyFrame>& toKeyFrame
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
