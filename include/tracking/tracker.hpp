#pragma warning(push, 0)
#include<opencv2/core.hpp>
#pragma warning(pop)

#include"calibration/calibration.hpp"
#include"frame/matcher.hpp"
#include"initializer.hpp"
#include"map/map.hpp"

namespace slam {

class Tracker {
public:
    enum States {
        NO_IMAGES = -1,
        UNINITIALIZED = 0,
        INITIALIZED = 1
    };
private:
    States state;
    std::shared_ptr<cv::Mat> cameraMatrix, distortions;

    std::shared_ptr<Frame> initialFrame, currentFrame;

    Initializer initializer;
    std::shared_ptr<Detector> detector;
    Matcher matcher;

public:
    std::shared_ptr<Map> map;
public:
    Tracker(Calibration calibration, std::shared_ptr<Detector> detector);
    /**
     * Initialize tracker from initial frame and current frame.
     * Successfull initialization creates initial map,
     * sets reference keyframe.
     */
    void track(std::shared_ptr<cv::Mat> image);
private:
    bool initialize();

    std::shared_ptr<Frame> packImage(
        std::shared_ptr<cv::Mat> image, double timestamp
    );
};

};
