#ifndef TRACKER_H
#define TRACKER_H

#pragma warning(push, 0)
#include<opencv2/core.hpp>
#pragma warning(pop)

#include"calibration/calibration.hpp"
#include"frame/matcher.hpp"
#include"initializer.hpp"
#include"map/map.hpp"
#include"tracking/mapper.hpp"

namespace slam {

/**
 * Class that performes tracking of the KeyPoints in the frames.
 */
class Tracker {
public:
    enum States {
        NO_IMAGES,
        UNINITIALIZED,
        INITIALIZED
    };
private:
    std::shared_ptr<cv::Mat> cameraMatrix, distortions;

    std::shared_ptr<KeyFrame> initialKeyFrame, lastKeyFrame, currentKeyFrame;

    Initializer initializer;
    std::shared_ptr<Detector> detector;
    Matcher matcher;
    Mapper mapper;

    cv::Mat velocity;
    bool useMotion;
public:
    States state;

    std::shared_ptr<Map> map;
public:
    Tracker(
        Calibration calibration,
        std::shared_ptr<Detector> detector,
        bool useMotion = true
    );
    /**
     * Initialize tracker from initial frame and current frame.
     * Successfull initialization creates initial map,
     * sets reference keyframe.
     */
    void track(std::shared_ptr<cv::Mat> image);
private:
    /**
     * Attempt to initialize tracking and map.
     * Successful initialization creates initial map from
     * `initialFrame` and `currentFrame`.
     *
     * @return `True` if initializetion was successful
     * (found enought matches for initial reconstruction).
     * `False` --- otherwise.
     */
    bool _initialize();

    /**
     * Track features from `last` KeyFrame onto `current` KeyFrame.
     * Tracking is done as follows:
     * - Find matches between `last` and `current` KeyFrame
     * - Optimize `current` KeyFrame pose using matches
     * - Find more matches, using optimized `current` KeyFrame pose,
     *   using Matcher.projectionMatch
     * - Optimize `current` KeyFrame pose again
     *
     * @return `True` if tracking was successful (found enought matches).
     * `False` --- otherwise.
     */
    bool _trackFrame();

    /**
     * Track features from `last` KeyFrame onto `current` KeyFrame
     * using motion model.
     * Motion tracking is done as follows:
     * - Calculate `current` KeyFrame pose using `last` KeyFrame pose and
     *   motion model
     * - Find matches between `last` and `current` KeyFrames using
     *   Matcher.projectionMatch
     * - Optimize `current` KeyFrame pose using matches
     *
     * @return `True` if tracking was successful (found enought matches).
     * `False` --- otherwise.
     */
    bool _trackMotionFrame();

    /**
     * If tracking was successful, then update motion model,
     * otherwise --- reset it.
     *
     * New motion model is calculated from `lastKeyFrame` and `currentKeyFrame`
     * poses as follows:
     *
     * \f[
     * \begin{align}
     * & M_{LF} = \left[ R_{LF}^T \ |\  -R_{LF}^T \cdot t_{LF} \right], \\
     * & V = P_{CF} \cdot M_{LF},
     * \end{align}
     * \f]
     *
     * where \f$ V \f$ --- is the new velocity (a.k.a. new motion model),\n
     * \f$ R_{LF} \f$ --- rotation of the `last` KeyFrame,\n
     * \f$ t_{LF} \f$ --- translation of the `last` KeyFrame,\n
     * \f$ | \f$ --- concatenation of rotation and translation matrices
     * to homogeneous one,\n
     * \f$ P_{CF} \f$ --- pose of the `current` KeyFrame.
     *
     * @param successfulTracking Whether or not tracking was successful.
     * If not --- motion model is set to empty.
     */
    void _updateMotion(bool successfulTracking);

    /**
     * Add mappoints to `keyframe` with `trainIdx` if there
     * is mappoint in `lastMappoints` under `queryIdx` key
     * from `matches`.
     */
    void _addMatches(
        std::shared_ptr<KeyFrame> keyframe,
        const std::map<int, std::shared_ptr<MapPoint>>& lastMappoints,
        const std::vector<cv::DMatch>& matches
    );

    /**
     * Create KeyFrame from `image` with identity pose matrix.
     *
     * @param image Image to pack into KeyFrame.
     * @param timestamp Timestamp of when this image was taken.
     * @return New KeyFrame with identity pose matrix and no mappoints.
     */
    std::shared_ptr<KeyFrame> _packImage(
        std::shared_ptr<cv::Mat> image, double timestamp
    );
};

};

#endif
