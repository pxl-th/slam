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

    double calibrationError;

public:
    Calibration() = default;
    Calibration(
        const CalibrationSettings& settings,
        bool display = true,
        float resize = 1080.0f
    );
    ~Calibration() = default;

    void read(const cv::FileNode& node);

    void write(cv::FileStorage& fs) const;

private:
    std::optional<std::tuple<std::vector<std::vector<cv::Point2f>>, cv::Size>>
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
