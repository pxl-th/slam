#pragma warning(push, 0)
#include<opencv2/core.hpp>
#pragma warning(pop)

#include"calibration/calibration.hpp"
#include"frame/matcher.hpp"
#include"initializer.hpp"
#include"map/map.hpp"

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
    States state;
    std::shared_ptr<cv::Mat> cameraMatrix, distortions;

    std::shared_ptr<KeyFrame> initialKeyFrame, lastKeyFrame, currentKeyFrame;

    Initializer initializer;
    std::shared_ptr<Detector> detector;
    Matcher matcher;

    cv::Mat velocity;
    bool useMotion;
public:
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
    bool _initialize();

    // TODO: comms
    void _trackFrame();

    void _trackMotionFrame();

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
     * Create KeyFrame from `image` with identity pose matrix.
     */
    std::shared_ptr<KeyFrame> _packImage(
        std::shared_ptr<cv::Mat> image, double timestamp
    );
};

};
