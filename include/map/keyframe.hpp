#ifndef KEYFRAME_H
#define KEYFRAME_H

#pragma warning(push, 0)
#include<map>
#include<vector>

#include<opencv2/core.hpp>
#pragma warning(pop)

#include"frame/frame.hpp"
#include"mappoint.hpp"

namespace slam {

class MapPoint;

/**
 * Holder for Frames and camera position info.
 */
class KeyFrame {
private:
    std::shared_ptr<Frame> frame;
    cv::Mat pose, cameraCenter;
public:
    /**
     * Map with `{keypoint id : mappoint}` elements,
     * where `keypoint id` is for the current frame's keypoint.
     */
    std::map<int, std::shared_ptr<MapPoint>> mappoints;
    /**
     * Map with `{keyframe : connections}` elements,
     * where `connections` is the number of mappoints that
     * are shared between `keyframe` and this KeyFrame.
     */
    std::map<std::shared_ptr<KeyFrame>, size_t> connections;
    static unsigned long long globalID;
    unsigned long long id;
public:
    KeyFrame(std::shared_ptr<Frame> frame, const cv::Mat& pose);

    cv::Mat getPose() const;
    cv::Mat getCameraCenter() const;
    std::shared_ptr<Frame> getFrame() const;
    /**
     * Given pose matrix, update KeyFrame's pose and camera's center.
     *
     * @param pose New pose matrix.
     * Should contain both rotation and translation,
     * and have `4x4` shape.
     */
    void setPose(const cv::Mat& newPose);

    void addMapPoint(int keypointId, std::shared_ptr<MapPoint> mapPoint);

    bool existMapPoint(int keypointId) const;

    void removeMapPoint(int keypointId);

    std::map<int, std::shared_ptr<MapPoint>> getMapPoints() const;

    size_t mappointsNumber() const;
    /**
     * Calculate median depth of the mappoints, visible in this keyframe,\n
     * by calculating depth for all visible mappoints and getting their median.\n
     * Depth for mappoint \f$ p_i \f$ is calculated as follows:\n
     *
     * \f$
     * d_i = T_d \cdot p_i + z, T_d = T[2, 0:3]^T, z = T[2, 3]
     * \f$
     *
     * where \f$ T \f$ --- keyframe's pose matrix,\n
     * \f$T_d\f$ --- extracted depth transformation from \f$ T \f$,\n
     * \f$ z \f$ --- depth translation.
     *
     * @return Median depth.
     */
    float medianDepth() const;
    float medianDepth(std::vector<std::shared_ptr<MapPoint>> points) const;
};

};

#endif
