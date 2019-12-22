#ifndef CALIBRATION_H
#define CALIBRATION_H

#pragma warning(push, 0)
#include<string>
#include<tuple>
#include<vector>

#include<opencv2/core.hpp>
#pragma warning(pop)

#include"calibration_settings.hpp"

namespace slam {

/**
 * Class that performs calibration procedure in order to extract
 * intrinsic and extrinsic parameters of the camera.
 * Supports only [chessboard](https://docs.opencv.org/master/d9/d0c/
 * group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a) pattern.
 *
 * Intrinsic parameters are:
 * - focal length
 * - image sensor format
 * - principal point
 * - lens distortion
 *
 * Extrinsic parameters are Rotation and Translation matrices,
 * which denote coordinate transformation from 3D world coordinate
 * to 3D camera coordinates.
 */
class Calibration {
public:
    cv::Mat cameraMatrix;
    cv::Mat distortions;

    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;

    double calibrationError;

public:
    Calibration() = default;
    /**
     * Given calibration settings perform calibration procedure.
     *
     * @param settings Settings to use for calibration.
     * @param display If `True` then for each image in settings
     * found pattern will be showed.
     * @param resize Resolution to which to resize image
     * before performing calibration.
     * If `-1`, then no resizing is done.
     */
    Calibration(
        const CalibrationSettings& settings,
        bool display = true,
        float resize = -1
    );
    ~Calibration() = default;

    void read(const cv::FileNode& node);

    void write(cv::FileStorage& fs) const;

private:
    std::tuple<std::vector<std::vector<cv::Point2f>>, cv::Size>
    _findPoints(
        const CalibrationSettings& settings,
        bool display,
        float resize
    );

    std::vector<std::vector<cv::Point3f>>
    _calculateBorderCornerPosition(const CalibrationSettings& settings);

    void _calibrate(
        const CalibrationSettings& settings,
        const std::vector<std::vector<cv::Point2f>>& points,
        const cv::Size& imageSize
    );
};

};

#endif
