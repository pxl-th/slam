#pragma warning(push, 0)
#include<queue>
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
    bool acceptKeyframes;

    std::shared_ptr<KeyFrame> currentKeyFrame;
    std::queue<std::shared_ptr<KeyFrame>> keyframeQueue;

    Matcher matcher;
public:
    std::shared_ptr<Map> map;
public:
    Mapper(std::shared_ptr<Map> map);

    /**
     * @return `True` if Mapper accepts new KeyFrame, `False` --- otherwise.
     */
    bool accepts() const;

    void addKeyframe(std::shared_ptr<KeyFrame> keyframe);
    /**
     * Triangulate keypoints between `keyframe1` and `keyframe2`.
     * *Note* that KeyFrame's have to potentially share keypoints,
     * otherwise triangulation will be incorrect.
     *
     * @return List of triangulated keypoints, pose matrix for `frame2`
     * and outliers mask (`0` --- is outlier).
     */
    static std::tuple<std::vector<cv::Point3f>, cv::Mat, cv::Mat>
    triangulatePoints(
        std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2,
        std::vector<cv::DMatch> matches
    );
private:
    void _processKeyFrame(std::shared_ptr<KeyFrame> keyframe);
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
};

};
