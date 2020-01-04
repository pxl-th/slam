#pragma warning(push, 0)
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

    _createConnections(currentKeyFrame);

    for (const auto& [keyframe, connections] : currentKeyFrame->connections) {
        auto matches = matcher.frameMatch(
            keyframe->getFrame(), currentKeyFrame->getFrame(), 300, 50
        );
        std::cout << "[mapping] Matches " << matches.size() << std::endl;
        if (matches.size() < 10) continue;
        auto points = std::get<1>(triangulatePoints(
            keyframe, currentKeyFrame, matches, false
        ));

        for (size_t i = 0; i < matches.size(); i++) {
            auto mappoint = std::make_shared<MapPoint>(
                points[i], currentKeyFrame
            );
            mappoint->addObservation(keyframe, matches[i].queryIdx);
            mappoint->addObservation(currentKeyFrame, matches[i].trainIdx);

            keyframe->addMapPoint(matches[i].queryIdx, mappoint);
            currentKeyFrame->addMapPoint(matches[i].trainIdx, mappoint);

            map->addMappoint(mappoint);
        }
    }
    map->addKeyframe(currentKeyFrame);
    std::cout
        << "[mapping] Mapped mappoints "
        << currentKeyFrame->getMapPoints().size() << std::endl;
    /**
     * + create connections between keyframes before adding to the map
     * + add kf to map
     * + triangulate points
     * + replace triangulation in initializer
     * + for every keyframe connection triangulate
     * - fuse duplicates
     * - perform BA
     * - remove outliers
     */
}

void Mapper::_createConnections(
    std::shared_ptr<KeyFrame> targetKeyFrame, int threshold
) {
    targetKeyFrame->connections.clear();
    std::map<std::shared_ptr<KeyFrame>, int> counter;
    // Count number of mappoints that are shared
    // between each KeyFrame and this KeyFrame.
    for (const auto& [id, mappoint] : targetKeyFrame->getMapPoints()) {
        for (const auto& [keyframe, keypointId] : mappoint->getObservations()) {
            if (keyframe->id == targetKeyFrame->id) continue;
            counter[keyframe]++;
        }
    }
    // Create connections with KeyFrames that more than `threshold`
    // shared MapPoints.
    int maxCount = 0;
    for (const auto& [keyframe, count] : counter) {
        if (count > maxCount) maxCount = count;
        if (count < threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
    // If no KeyFrame's have been found --- add KeyFrame with maximum count.
    if (targetKeyFrame->connections.empty()) threshold = maxCount;
    for (const auto& [keyframe, count] : counter) {
        if (count != threshold) continue;
        targetKeyFrame->connections[keyframe] = count;
        keyframe->connections[targetKeyFrame] = count;
    }
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
    cv::Mat firstProjection(3, 4, CV_32F), secondProjection(3, 4, CV_32F);
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

};
