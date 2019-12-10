#ifndef CALIBRATION_H
#define CALIBRATION_H

#include<string>
#include<optional>
#include<tuple>
#include<vector>

#include<opencv2/core.hpp>

#include"calibration_settings.hpp"

namespace slam {

class Calibration {
public:
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;

    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;

    std::vector<float> reprojectionErrors;
    double averageError;

public:
    Calibration(const CalibrationSettings& settings, bool display = true);
    ~Calibration() = default;

private:
    std::optional<std::tuple<std::vector<std::vector<cv::Point2f>>, cv::Size>>
    _findPoints(const CalibrationSettings& settings, bool display);

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
