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
     * @param keyframe KeyFrame for which to create MapPoints.
     * @param threshold Minimum number of MapPoint two KeyFrames
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
    /**
     * Find MapPoint duplicates in the map and fuse them into one
     * thus removing duplicates.
     *
     * @param keyframes Number of most recently added to `map` KeyFrame's
     * to check for duplicates.
     * @param connections Number of connection KeyFrames to consider,
     * for a current KeyFrame, when finding duplicates.
     */
    void _fuseDuplicates(const int keyframes = 3, const int connections = 3);
    /**
     * Find MapPoint duplicates between two KeyFrames.
     *
     * @param keyframe1 First keyframe for which to find duplicates.
     * @param keyframe2 Second keyframe in which to find duplicates.
     */
    void _keyframeDuplicates(
        std::shared_ptr<KeyFrame>& keyframe1, std::shared_ptr<KeyFrame>& keyframe2
    );
    /**
     * Given two MapPoints test, whether they are duplicates of each other.
     *
     * Test involves:\n
     * 1) checking if they correspond to the same feature point;\n
     * 2) checking if their descriptors are closer than `descriptorDistance`
     * in terms of Hamming distance;\n
     * 3) checking if the distance between MapPoints in space is smaller
     * than `pointDistance` \f$ \left\lVert p_1 - p_2 \right\rVert < d \f$,
     * where \f$d\f$ --- `pointDistance`.
     *
     * If at least two of the three test pass --- mappoints are
     * considered to be outliers.
     *
     * @param mappoint1 First MapPoint.
     * @param feature1 Id of the feature to which first MapPoint corresponds.
     * @param mappoint2 Second MapPoint.
     * @param feature1 Id of the feature to which second MapPoint corresponds.
     * @param descriptorDistance Descriptors with distance between them
     * lower than this threshold (in terms of Hamming distance)
     * are considered to be duplicates.
     * @param pointDistance MapPoints with distance between them lower
     * than this threshold (in terms of L2 metric)
     * are considered to be duplicates.
     * @return `true` if two MapPoints are duplicates of each other,
     * `false` --- otherwise.
     */
    bool _isDuplicate(
        const std::shared_ptr<MapPoint>& mappoint1, const int feature1,
        const std::shared_ptr<MapPoint>& mappoint2, const int feature2,
        const int descriptorDistance = 50, const double pointDistance = 1E-2
    ) const;
};

};

#endif
