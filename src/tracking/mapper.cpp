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
}

void Mapper::clearQueue() {
    keyframeQueue = std::queue<std::shared_ptr<KeyFrame>>();
}

bool Mapper::initialize() {
    if (keyframeQueue.empty() || keyframeQueue.size() < 2)
        return false;

    auto initial = keyframeQueue.front(); keyframeQueue.pop();
    current = keyframeQueue.front(); keyframeQueue.pop();

    auto matches = matcher.frameMatch(initial, current, {}, 300, -1, 4);
    if (matches.size() < 100) return false;
    auto [reconstructedPoints, pose, mask] = std::get<0>(triangulatePoints(
        initial, current, matches, true
    ));

    std::vector<std::shared_ptr<MapPoint>> mappoints;
    map = std::make_shared<Map>();
    initial->setPose(cv::Mat::eye(4, 4, CV_32F));
    current->setPose(pose);
    // Construct mappoints omitting outliers.
    for (size_t i = 0, j = 0; i < matches.size(); i++) {
        if (mask.at<uchar>(static_cast<int>(i)) == 0) continue;
        auto point = reconstructedPoints[j++];
        if (isOutlier(point, initial, current, matches[i]))
            continue;

        auto mappoint = std::make_shared<MapPoint>(point, current);
        mappoint->addObservation(initial, matches[i].queryIdx);
        mappoint->addObservation(current, matches[i].trainIdx);

        initial->addMapPoint(matches[i].queryIdx, mappoint);
        current->addMapPoint(matches[i].trainIdx, mappoint);

        mappoints.push_back(mappoint);
    }
    if (mappoints.empty()) return false;

    float inverseDepth = 1.0f / initial->medianDepth();
    std::cout << "[initialization] Inverse depth " << inverseDepth << std::endl;
    if (inverseDepth < 0) return false;
    // Scale translation by inverse median depth.
    auto currentPose = current->getPose();
    currentPose.col(3).rowRange(0, 3) = (
        currentPose.col(3).rowRange(0, 3) * inverseDepth
    );
    current->setPose(currentPose);
    // Scale mappoints by inverse median depth.
    for (auto& [id, p] : initial->getMapPoints())
        p->setWorldPos(p->getWorldPos() * inverseDepth);
    // Add KeyFrames and MapPoints to map.
    map->addKeyframe(initial);
    map->addKeyframe(current);
    for (auto mappoint : mappoints) map->addMappoint(mappoint);
    optimizer::globalBundleAdjustment(map, 20);
    return true;
}

void Mapper::process() {
    lastReconstruction++;
    current = keyframeQueue.front();
    keyframeQueue.pop();
    std::cout << "[mapping] KeyFrame id " << current->id << std::endl;
    // For current KeyFrame create connections with other KeyFrames,
    // that share enough mappoints with it.
    _createConnections(current, 0.2f * current->mappointsNumber());
    // Try sharing already existing MapPoints with new KeyFrame.
    // If enough MapPoints were shared, then there is no need to create new.
    if (_share(current) && lastReconstruction < 4) {
        // Enough MapPoints were shared, no need to create new ones.
        map->addKeyframe(current);
        std::cout
            << "[sharing] Map contains " << map->getMappoints().size()
            << " mappoints" << std::endl;
        std::cout
            << "[sharing] Enought mappoints were shared: "
            << current->mappointsNumber() << std::endl;
        return;
    }
    std::cout << "[mapping] Reconstruction" << std::endl;
    lastReconstruction = 0;
    // For each connection triangulate matches and add
    // new MapPoints to the map if they pass outliers test.
    for (auto& connection : current->connections) {
        // If enough MapPoints, add no more.
        if (current->mappointsNumber() >= 200) break;
        // Otherwise, add some more.
        auto keyframe = connection.first;
        auto matches = matcher.inverseMappointsFrameMatch(current, keyframe);
        if (matches.size() < 10) continue;
        auto points = std::get<1>(triangulatePoints(keyframe, current, matches, false));

        size_t inliers = 0;
        for (size_t i = 0; i < matches.size(); i++) {
            auto point = points[i]; auto match = matches[i];
            if (isOutlier(point, keyframe, current, match))
                continue;

            ++inliers;
            auto mappoint = std::make_shared<MapPoint>(point, current);

            mappoint->addObservation(keyframe, match.queryIdx);
            mappoint->addObservation(current, match.trainIdx);

            keyframe->addMapPoint(match.queryIdx, mappoint);
            current->addMapPoint(match.trainIdx, mappoint);

            map->addMappoint(mappoint);
        }
        if (inliers > 0) _keyframeDuplicates(current, keyframe);
    }
    map->addKeyframe(current);
    /* _removeDuplicates(); */
    std::cout
        << "[mapping] Added keyframe with "
        << current->mappoints.size() << " mappoints" << std::endl;
}

void Mapper::_createConnections(
    std::shared_ptr<KeyFrame> targetKeyFrame, int threshold
) {
    if (!targetKeyFrame->connections.empty())
        targetKeyFrame->connections.clear();
    std::map<std::shared_ptr<KeyFrame>, int> counter;
    std::cout
        << "[mapping] Mappoints for connections "
        << targetKeyFrame->mappointsNumber() << std::endl;
    assert(targetKeyFrame->mappointsNumber() != 0);
    // Count number of mappoints that are shared
    // between each KeyFrame and this KeyFrame.
    int maxCount = 0;
    for (const auto [id, mappoint] : targetKeyFrame->getMapPoints()) {
        for (const auto [keyframe, keypointId] : mappoint->getObservations()) {
            if (keyframe->id == targetKeyFrame->id) continue;
            int count = ++counter[keyframe];
            if (count > maxCount) maxCount = count;
        }
    }
    threshold = std::min(threshold, maxCount);
    std::cout << "[mapping] Connection threshold " << threshold << std::endl;
    // Create connections with KeyFrames that more than `threshold`
    // shared MapPoints.
    for (const auto [keyframe, count] : counter) {
        if (count < threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
    }
    std::cout << "[mapping] Connections" << std::endl;
    for (const auto& [k, c] : targetKeyFrame->connections)
        std::cout << k << ": " << c << std::endl;
}

bool Mapper::_share(std::shared_ptr<KeyFrame>& keyframe, float matchRelation) {
    std::unordered_set<unsigned long long> sharedMappoints;
    // Find matches between `keyframe` and its connections
    // taking into account connection's MapPoints.
    for (const auto& [connection, count] : current->connections) {
        auto matches = matcher.mappointsFrameMatch(connection, current);
        if (matches.size() < matchRelation * connection->mappointsNumber())
            continue;
        // Enough matches, add MapPoints from connection KeyFrame.
        for (const auto& match : matches) {
            if (current->mappoints.find(match.trainIdx) != current->mappoints.end())
                continue;
            auto& mappoint = connection->mappoints[match.queryIdx];
            if (mappoint->observations.find(current) != mappoint->observations.end())
                continue;
            if (sharedMappoints.find(mappoint->id) != sharedMappoints.end())
                continue;
            current->addMapPoint(match.trainIdx, mappoint);
            mappoint->addObservation(current, match.trainIdx);
            sharedMappoints.insert(mappoint->id);
        }
    }
    return keyframe->mappointsNumber() >= 100;
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

void Mapper::_removeDuplicates(const int keyframes, const int connections) {
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
}

void Mapper::_keyframeDuplicates(
    std::shared_ptr<KeyFrame>& keyframe1, std::shared_ptr<KeyFrame>& keyframe2
) {
    cv::Mat descriptor1, descriptor2;
    std::vector<std::shared_ptr<MapPoint>> duplicates;
    std::vector<std::tuple<
        std::shared_ptr<MapPoint>, std::shared_ptr<KeyFrame>, int
    >> replacements;
    // Find MapPoint duplicate candidates.
    for (auto [id1, mappoint1] : keyframe1->mappoints) {
        descriptor1 = keyframe1->getFrame()->descriptors->row(id1);
        for (auto [id2, mappoint2] : keyframe2->mappoints) {
            if (mappoint1 == mappoint2) continue;
            descriptor2 = keyframe2->getFrame()->descriptors->row(id2);
            if (!_isDuplicate(mappoint1, descriptor1, mappoint2, descriptor2))
                continue;
            // TODO: retain one with smaller reprojection error
            duplicates.push_back(mappoint2);
            replacements.push_back({mappoint1, keyframe2, id2});
        }
    }
    // Remove duplicates.
    for (auto duplicate : duplicates) {
        map->removeMappoint(duplicate);
        for (auto [keyframe, id] : duplicate->getObservations())
            keyframe->removeMapPoint(id);
    }
    // Add remained mappoints to the KeyFrames
    // from which duplicates were removed.
    for (auto [mappoint, keyframe, id] : replacements) {
        // TODO check if replacement was not in duplicates previously
        mappoint->addObservation(keyframe, id);
        keyframe->addMapPoint(id, mappoint);
    }
}

bool Mapper::_isDuplicate(
    const std::shared_ptr<MapPoint>& mappoint1, const cv::Mat& descriptor1,
    const std::shared_ptr<MapPoint>& mappoint2, const cv::Mat& descriptor2,
    const int descriptorDistance, const double pointDistance
) const {
    bool closeDescriptors, closePoints;
    std::vector<cv::DMatch> match;

    matcher.matcher->match(descriptor1, descriptor2, match);
    closeDescriptors = (
        (!match.empty()) && match[0].distance <= descriptorDistance
    );
    if (!closeDescriptors) return false;
    closePoints = (
        cv::norm(mappoint1->getWorldPos() - mappoint2->getWorldPos())
        < pointDistance
    );
    return closePoints;
}

};
