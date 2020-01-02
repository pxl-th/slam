#ifndef MAPPER_H
#define MAPPER_H

#pragma warning(push, 0)
#include<queue>
#include<variant>
#pragma warning(pop)

#include"frame/matcher.hpp"
#include"map/map.hpp"

namespace slam {

/**
 * Mapper manages expansion of Map initialized by Tracker.
 *
 * Creates new MapPoint's from KeyFrame's and refines Map.
 */
class Mapper {
private:
    std::shared_ptr<KeyFrame> currentKeyFrame;
    std::queue<std::shared_ptr<KeyFrame>> keyframeQueue;

    Matcher matcher;
public:
    std::shared_ptr<Map> map;
public:
    Mapper();
    Mapper(Matcher matcher);

    void addKeyframe(std::shared_ptr<KeyFrame> keyframe);
    /**
     * TODO
     */
    static std::variant<
        std::tuple<std::vector<cv::Point3f>, cv::Mat, cv::Mat>,
        std::vector<cv::Point3f>
    > triangulatePoints(
        std::shared_ptr<KeyFrame> keyframe1, std::shared_ptr<KeyFrame> keyframe2,
        std::vector<cv::DMatch> matches, bool recoverPose
    );
private:
    void _processKeyFrame();
    /**
     * Create connections between keyframes that
     * share `mappoints` with `keyframe`.
     *
     * @param keyframe KeyFrame for which to create MapPoint's.
     * @param threshold Minimum number of MapPoint two KeyFrame's
     * have to share to form a connection.
     */
    void _createConnections(
        std::shared_ptr<KeyFrame> keyframe, int threshold = 15
    );

    static std::tuple<cv::Mat, cv::Mat> _recoverPose(
        std::vector<cv::Point2f>& frame1Points,
        std::vector<cv::Point2f>& frame2Points,
        const cv::Mat& cameraMatrix
    );
};

};

#endif
