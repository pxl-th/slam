#ifndef MAPPOINT_H
#define MAPPOINT_H

#pragma warning(push, 0)
#include<map>
#pragma warning(pop)

#include"keyframe.hpp"

namespace slam {

class KeyFrame;

/**
 * Mappoint contains positions of the triangulated KeyPoint in 3D space.
 */
class MapPoint {
private:
    /**
     * Position of the MapPoint in space.
     */
    cv::Point3f position;
    /**
     * Main KeyFrame from which this MapPoint was created.
     */
    std::shared_ptr<KeyFrame> keyframe;
    /**
     * Keyframe from which mappoint is visible
     * and id of the keypoint of the mappoint.
     */
    std::map<std::shared_ptr<KeyFrame>, int> observations;
public:
    MapPoint(const cv::Point3f& position, std::shared_ptr<KeyFrame> keyframe);
    /**
     * Get KeyFrame from which this MapPoint was created.
     */
    std::shared_ptr<KeyFrame> getReferenceKeyframe() const;
    /**
     * Get position of MapPoint in space.
     * @return World coordinates of the MapPoint.
     */
    cv::Point3f getWorldPos() const;
    /**
     * Set position of MapPoint in space in world coordinates.
     * @param New world coordinates of the MapPoint.
     */
    void setWorldPos(const cv::Point3f& newPos);
    /**
     * Get observations for the MapPoint.
     * @return Mapping `{keyframe : id}`.
     * `keyframe` from which mappoint is visible
     * and `id` of the keypoint in that `keyframe` of the mappoint.
     */
    std::map<std::shared_ptr<KeyFrame>, int> getObservations() const;
    /**
     * Add new observation to the MapPoint.
     * @param keyframeO KeyFrame which observes this MapPoint.
     * @param id Id of the keypoint in `keyframeO`
     * which corresponds to this MapPoint.
     */
    void addObservation(std::shared_ptr<KeyFrame> keyframeO, int id);
    /**
     * Calculate parallax for point.
     *
     * \f[
     * \begin{align}
     * & n_1 = p - c_1 \\
     * & n_2 = p - c_2 \\
     * & l = \frac
     *  {n_1 \cdot n_2}
     *  {\left\lVert n_1 \right\rVert \cdot \left\lVert n_2 \right\rVert} \\
     * & g = acos\left( l \right) \cdot 180 \cdot \frac{1}{\pi}
     * \end{align}
     * \f]
     *
     * where \f$ p \f$ --- position of point in space,\n
     * \f$ c_1, c_2 \f$ --- positions of cameras,\n
     * \f$ l \f$ --- cosine parallax and \f$ g \f$ --- parallax in degrees.
     *
     * @param point Position of point for which to calculate parallax.
     * @param camera1 Position of the first camera that observes the `point`.
     * @param camera2 Position of the second camera that observes the `point`.
     * @return Parallax value in degrees.
     */
    static double parallax(cv::Mat point, cv::Mat camera1, cv::Mat camera2);
};

};

#endif
