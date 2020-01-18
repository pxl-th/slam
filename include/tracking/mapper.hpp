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
    std::shared_ptr<KeyFrame> current;
    std::queue<std::shared_ptr<KeyFrame>> keyframeQueue;
    int lastReconstruction = 0;

    Matcher matcher;
public:
    std::shared_ptr<Map> map;
public:
    Mapper();
    Mapper(Matcher matcher);
    /**
     * Add KeyFrame to queue for processing.
     *
     * @param keyframe KeyFrame to process and add to the map.
     */
    void addKeyframe(std::shared_ptr<KeyFrame> keyframe);
    /**
     * Clear content of the keyframe queue.
     */
    void clearQueue();
    /**
     * Initialize map.
     * For this, keyframe queue should have two KeyFrames
     * from which to create map.
     *
     * @return `true` if successfully initialized map, `false` --- otherwise.
     */
    bool initialize();
    void process();
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
    /**
     * Create connections between keyframes that
     * share `mappoints` with `keyframe`.
     *
     * @param keyframe KeyFrame for which to create MapPoints.
     * @param threshold Minimum number of MapPoint two KeyFrames
     * have to share to form a connection.
     */
    void _createConnections(
        std::shared_ptr<KeyFrame> keyframe, size_t threshold = 15
    );
    /**
     * Share existing MapPoints in `keyframe` connections
     * to facilitate MapPoint reusability.
     *
     * @param keyframe KeyFrame with which MapPoints will be shared.
     * @param matchRelation Match relation for `keyframe` and its connection.
     * If amount of matches is greater than
     * `matchRelation * connection.mappointsNumber()` then MapPoints of
     * this connection will be shared with `keyframe`.
     * @return `true` if enough MapPoints were shared with `keyframe`
     * among all its connections. `false` --- otherwise.
     */
    bool _share(std::shared_ptr<KeyFrame>& keyframe, float matchRelation = 0.3f);
    /**
     * TODO doc
     */
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
    void _removeDuplicates(const int keyframes = 3, const int connections = 3);
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
     * 1) checking if their descriptors are closer than `descriptorDistance`
     * in terms of Hamming distance;\n
     * 2) checking if the distance between MapPoints in space is smaller
     * than `pointDistance` \f$ \left\lVert p_1 - p_2 \right\rVert < d \f$,
     * where \f$d\f$ --- `pointDistance`.
     *
     * If both tests pass --- then these points considered to be outliers.
     *
     * @param mappoint1 First MapPoint.
     * @param descriptor1 Descriptor of the KeyPoint from which
     * first MapPoint was created.
     * @param mappoint2 Second MapPoint.
     * @param descriptor2 Descriptor of the KeyPoint from which
     * second MapPoint was created.
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
        const std::shared_ptr<MapPoint>& mappoint1, const cv::Mat& descriptor1,
        const std::shared_ptr<MapPoint>& mappoint2, const cv::Mat& descriptor2,
        const int descriptorDistance = 100, const double pointDistance = 1E-2
    ) const;
};

};

#endif
