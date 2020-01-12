#ifndef FRAME_H
#define FRAME_H

#pragma warning(push, 0)
#include<vector>

#include<opencv2/core.hpp>
#pragma warning(pop)

#include"detector.hpp"

namespace slam {

/**
 * Frame class that contains functions for holding and retrieving
 * info (e.g. keypoints) from the frame.
 */
class Frame {
public:
    std::shared_ptr<Detector> detector;
    std::shared_ptr<cv::Mat> image, cameraMatrix, distortions, descriptors;
    std::vector<cv::KeyPoint> keypoints, undistortedKeypoints;
    /**
     * Array of scales, size equals to the number of levels
     * in the `detector`'s pyramid.
     * At each level, \f$ s_i = s_{i - 1} * p_s \f$,
     * where \f$ s_i \f$ --- is the scale value at \f$ i \f$-th level,
     * \f$ p_s \f$ --- is the pyramid's scale value.
     */
    std::vector<float> scales;
    /**
     * Array with the same size as the number of levels
     * in the `detector`'s pyramid.
     * \f$ \sigma_i = s_i^2 \f$, where \f$ s_i \f$ --- is the scale
     * value at the \f$ i \f$-th level.
     */
    std::vector<float> sigma;
    /**
     * Array of inverted values of the sigmas \f$ \frac{1}{\sigma_i} \f$.
     * These values are used as information for the edges
     * of the hypergraph used in the Bundle Adjustment.
     */
    std::vector<float> invSigma;
public:
    Frame() {}
    /**
     * Create frame from given image.
     *
     * @param image Image of the current frame.
     * @param detector Keypoints detector for feature extraction.
     * @param cameraMatrix Camera matrix calculated using Calibration.
     * @param distortions Distortions coefficients calculated using Calibration.
     */
    Frame(
        std::shared_ptr<cv::Mat> image,
        std::shared_ptr<Detector> detector,
        std::shared_ptr<cv::Mat> cameraMatrix,
        std::shared_ptr<cv::Mat> distortions
    );
private:
    void _undistortKeyPoints();
};

};

#endif
