#pragma warning(push, 0)
#include<queue>
#pragma warning(pop)

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
public:
    std::shared_ptr<Map> map;
public:
    Mapper(std::shared_ptr<Map> map);

    /**
     * @return `True` if Mapper accepts new KeyFrame, `False` --- otherwise.
     */
    bool accepts() const;

    void addKeyframe(std::shared_ptr<KeyFrame> keyframe);
private:
    void _processKeyFrame(std::shared_ptr<KeyFrame> keyframe);
    /**
     * Create connections between keyframes that
     * have same `mappoints` visible as `keyframe`.
     */
    void _createConnections(
        std::shared_ptr<KeyFrame> keyframe, int threshold = 15
    );
};

};
