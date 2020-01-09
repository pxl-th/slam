#pragma warning(push, 0)
#include<algorithm>
#include<unordered_set>

#include<opencv2/core.hpp>
#include<opencv2/calib3d.hpp>
#pragma warning(pop)

#include"converter.hpp"
#include"tracking/mapper.hpp"
#include"tracking/optimizer.hpp"

namespace slam {

Mapper::Mapper() {}

Mapper::Mapper(Matcher matcher) : matcher(matcher) {}

void Mapper::addKeyframe(std::shared_ptr<KeyFrame> keyframe) {
    keyframeQueue.push(keyframe);
    _processKeyFrame();
}

void Mapper::_processKeyFrame() {
    currentKeyFrame = keyframeQueue.front();
    keyframeQueue.pop();
    std::cout << "[mapping] KeyFrame id " << currentKeyFrame->id << std::endl;
    // For current KeyFrame create connections with other KeyFrames,
    // that share enough mappoints with it.
    _createConnections(currentKeyFrame);
    auto currentCenter = currentKeyFrame->getCameraCenter();
    // For each connection triangulate matches and add
    // new MapPoints to the map if they pass outliers test.
    for (std::pair<std::shared_ptr<KeyFrame>, int> connection : currentKeyFrame->connections) {
        auto keyframe = connection.first;
        auto matches = matcher.frameMatch(
            keyframe->getFrame(), currentKeyFrame->getFrame(), 300, 50
        );
        if (matches.size() < 10) continue;

        std::cout
            << "[mapping] keyframe " << keyframe->id
            << " | matches " << matches.size() << std::endl;
        auto points = std::get<1>(triangulatePoints(
            keyframe, currentKeyFrame, matches, false
        ));
        int no = 0;

        for (size_t i = 0; i < matches.size(); i++) {
            auto point = points[i]; auto match = matches[i];
            if (isOutlier(point, keyframe, currentKeyFrame, match)) {
                no++;
                continue;
            }

            auto mappoint = std::make_shared<MapPoint>(point, currentKeyFrame);

            mappoint->addObservation(keyframe, match.queryIdx);
            mappoint->addObservation(currentKeyFrame, match.trainIdx);

            keyframe->addMapPoint(match.queryIdx, mappoint);
            currentKeyFrame->addMapPoint(match.trainIdx, mappoint);

            map->addMappoint(mappoint);
        }
        _keyframeDuplicates(currentKeyFrame, keyframe);

        std::cout << "Total outliers " << no << std::endl;
    }
    map->addKeyframe(currentKeyFrame);

    std::cout
        << "[mapping] Mapped mappoints before fusion "
        << currentKeyFrame->mappointsNumber() << std::endl;

    _fuseDuplicates();
    std::cout
        << "[mapping] Mapped mappoints "
        << currentKeyFrame->mappointsNumber() << std::endl;
    std::cout
        << "[mapping] Total keyframes in map "
        << map->getKeyframes().size() << std::endl;

    /* optimizer::globalBundleAdjustment(map); */
    /**
     * - perform BA
     */
}

void Mapper::_createConnections(
    std::shared_ptr<KeyFrame> targetKeyFrame, int threshold
) {
    targetKeyFrame->connections.clear();
    std::map<std::shared_ptr<KeyFrame>, int> counter;
    std::cout
        << "[mapping] Mappoints for connections "
        << targetKeyFrame->mappointsNumber() << std::endl;
    if (targetKeyFrame->mappointsNumber() == 0) assert(false);
    // Count number of mappoints that are shared
    // between each KeyFrame and this KeyFrame.
    for (const auto [id, mappoint] : targetKeyFrame->getMapPoints()) {
        for (const auto [keyframe, keypointId] : mappoint->getObservations()) {
            if (keyframe->id == targetKeyFrame->id) continue;
            if (counter.find(keyframe) == counter.end()) counter[keyframe] = 1;
            else counter[keyframe]++;
        }
    }
    std::cout << "[mapping] Counters" << std::endl;
    for (auto [k, c] : counter)
        std::cout << k << " | " << k->id << " | " << c << std::endl;
    // Create connections with KeyFrames that more than `threshold`
    // shared MapPoints.
    int maxCount = 0;
    for (const auto [keyframe, count] : counter) {
        if (count > maxCount) maxCount = count;
        if (count < threshold) continue;
        std::cout << keyframe->id << " | " << count << std::endl;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
    // If no KeyFrame's have been found --- add KeyFrame with maximum count.
    if (targetKeyFrame->connections.empty()) {
        threshold = maxCount;
        for (const auto [keyframe, count] : counter) {
            if (count < maxCount) continue;
            targetKeyFrame->connections[keyframe] = count;
            keyframe->connections[targetKeyFrame] = count;
        }
    }
    std::cout
        << "[mapping] Connections formed "
        << targetKeyFrame->connections.size() << std::endl;
}

std::variant<
    std::tuple<std::vector<cv::Point3f>, cv::Mat, cv::Mat>,
    std::vector<cv::Point3f>
> Mapper::triangulatePoints(
    std::shared_ptr<KeyFrame> keyframe1, std::shared_ptr<KeyFrame> keyframe2,
    std::vector<cv::DMatch> matches, bool recoverPose
) {
    auto frame1 = keyframe1->getFrame(), frame2 = keyframe2->getFrame();
    cv::Mat cameraMatrix = *frame1->cameraMatrix;
    // Copy points from keypoints.
    std::vector<cv::Point2f> frame1Points, frame2Points;
    frame1Points.resize(matches.size());
    frame2Points.resize(matches.size());
    for (size_t i = 0; i < matches.size(); i++) {
        frame1Points[i] = frame1->undistortedKeypoints[matches[i].queryIdx].pt;
        frame2Points[i] = frame2->undistortedKeypoints[matches[i].trainIdx].pt;
    }
    // Construct projection matrices.
    cv::Mat pose, mask;
    cv::Mat firstProjection = cv::Mat::zeros(3, 4, CV_32F);
    cv::Mat secondProjection = cv::Mat::zeros(3, 4, CV_32F);
    if (recoverPose) {
        std::tie(pose, mask) = _recoverPose(frame1Points, frame2Points, cameraMatrix);
        cameraMatrix.copyTo(firstProjection.rowRange(0, 3).colRange(0, 3));
        pose.rowRange(0, 3).colRange(0, 4).copyTo(secondProjection);
        secondProjection = cameraMatrix * secondProjection;
    } else {
        keyframe1->getPose().rowRange(0, 3).colRange(0, 4).copyTo(firstProjection);
        keyframe2->getPose().rowRange(0, 3).colRange(0, 4).copyTo(secondProjection);
        firstProjection = cameraMatrix * firstProjection;
        secondProjection = cameraMatrix * secondProjection;
    }
    // Reconstruct points.
    cv::Mat reconstructedPointsM, homogeneousPoints;
    cv::triangulatePoints(
        firstProjection, secondProjection,
        frame1Points, frame2Points, homogeneousPoints
    );
    cv::convertPointsFromHomogeneous(homogeneousPoints.t(), reconstructedPointsM);
    auto reconstructedPoints = vectorFromMat(reconstructedPointsM);

    if (recoverPose) return std::tuple{reconstructedPoints, pose, mask};
    return reconstructedPoints;
}

std::tuple<cv::Mat, cv::Mat> Mapper::_recoverPose(
    std::vector<cv::Point2f>& frame1Points,
    std::vector<cv::Point2f>& frame2Points,
    const cv::Mat& cameraMatrix
) {
    cv::Mat mask, essential = cv::findEssentialMat(
        frame1Points, frame2Points, cameraMatrix,
        cv::RANSAC, 0.999, 1.0, mask
    );
    cv::Mat rotation, translation, inliersMask;
    cv::recoverPose(
        essential, frame1Points, frame2Points,
        cameraMatrix, rotation, translation, inliersMask
    );
    // Complete outliers `mask` and remove outlier points.
    std::vector<cv::Point2f> rp, cp;
    for (int i = 0; i < inliersMask.rows; i++) {
        uchar im = inliersMask.at<uchar>(i), m = mask.at<uchar>(i);
        if (im == 0 || m == 0) {
            if (im == 0) mask.at<uchar>(i) = 0;
            continue;
        }
        rp.push_back(frame1Points[i]);
        cp.push_back(frame2Points[i]);
    }
    frame1Points = rp; frame2Points = cp;
    // Compose rotation and translation into pose matrix.
    cv::Mat pose = cv::Mat::eye(4, 4, CV_32F);
    rotation.copyTo(pose.rowRange(0, 3).colRange(0, 3));
    translation.copyTo(pose.rowRange(0, 3).col(3));

    return {pose, mask};
}

void Mapper::_fuseDuplicates(const int keyframes, const int connections) {
    size_t mp = map->getMappoints().size();

    size_t startIdx = std::max(
        static_cast<int>(map->getKeyframes().size()) - keyframes, 0
    );
    for (size_t i = startIdx; i < map->getKeyframes().size(); i++) {
        auto keyframe = map->getKeyframes()[i];
        int connectionId = 0;
        for (auto& connection : keyframe->connections) {
            if (connectionId++ == connections) break;
            std::shared_ptr<KeyFrame> connectionK = std::get<0>(connection);
            _keyframeDuplicates(keyframe, connectionK);
        }
    }
    std::cout
        << "Duplicates removed from map "
        << (mp - map->getMappoints().size()) << std::endl;
}

void Mapper::_keyframeDuplicates(
    std::shared_ptr<KeyFrame>& keyframe1, std::shared_ptr<KeyFrame>& keyframe2
) {
    std::unordered_set<std::shared_ptr<MapPoint>> duplicates;
    std::vector<std::tuple<
        std::shared_ptr<MapPoint>, std::shared_ptr<KeyFrame>, int
    >> replacements;
    // Find MapPoint duplicate candidates.
    for (auto& [id1, mappoint1] : keyframe1->mappoints) {
        for (auto& [id2, mappoint2] : keyframe2->mappoints) {
            if (!_isDuplicate(mappoint1, id1, mappoint2, id2))
                continue;
            // TODO: retain one with smaller reprojection error
            duplicates.insert(mappoint2);
            replacements.push_back({mappoint1, keyframe2, id2});
        }
    }
    // Remove duplicates.
    for (auto duplicate : duplicates) {
        map->removeMappoint(duplicate);
        for (auto& [keyframe, id] : duplicate->getObservations())
            keyframe->removeMapPoint(id);
    }
    // Add remained mappoints to the KeyFrames
    // from which duplicates were removed.
    for (auto& [mappoint, keyframe, id] : replacements) {
        if (duplicates.find(mappoint) != duplicates.end())
            continue;
        mappoint->addObservation(keyframe, id);
        keyframe->addMapPoint(id, mappoint);
    }
}

bool Mapper::_isDuplicate(
    const std::shared_ptr<MapPoint>& mappoint1, const int feature1,
    const std::shared_ptr<MapPoint>& mappoint2, const int feature2,
    const int descriptorDistance, const double pointDistance
) const {
    bool sameFeatures, closeDescriptors, closePoints;
    std::vector<cv::DMatch> match;

    sameFeatures = (
        feature1 == feature2
        && mappoint1->keyframe->id == mappoint2->keyframe->id
    );
    matcher.matcher->match(
        mappoint1->keyframe->getFrame()->descriptors->row(feature1),
        mappoint2->keyframe->getFrame()->descriptors->row(feature2),
        match
    );
    closeDescriptors = (
        (match.size() > 0) && match[0].distance <= descriptorDistance
    );
    if (sameFeatures && closeDescriptors) return true;
    if (sameFeatures || closeDescriptors) {
        closePoints = (
            cv::norm(mappoint1->getWorldPos() - mappoint2->getWorldPos())
            < pointDistance
        );
        if (closePoints) return true;
    }
    return false;
}

};
