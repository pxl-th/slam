#ifndef CALIBRATION_H
#define CALIBRATION_H

#include<string>
#include<optional>
#include<tuple>
#include<vector>

#include<opencv2/core.hpp>

namespace slam {

class CalibrationSettings {
public:
    cv::Size boardSize;
    float squareSize;

    std::vector<std::string> images;
    size_t currentImageId = 0;

public:
    CalibrationSettings();
    ~CalibrationSettings() = default;

    void read(const cv::FileNode& node);

    void write(cv::FileStorage& fs) const;
};

class CalibrationResult {
public:
    cv::Mat cameraMatrix;
    cv::Mat distortionCoefficients;

    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;

    std::vector<float> reprojectionErrors;
    double averageError;

public:
    CalibrationResult() = default;
    ~CalibrationResult() = default;
};

void write(
    cv::FileStorage& fs,
    const std::string&,
    const CalibrationSettings& settings
);

void read(
    const cv::FileNode& node,
    CalibrationSettings& settings,
    const CalibrationSettings& defaultSettings = CalibrationSettings()
);

CalibrationSettings loadCalibrationSettings(const std::string& settingsFile);

void calibrateCamera(
    const CalibrationSettings& settings, const std::string& outputFile,
    bool display = true
);

std::optional<std::tuple<
    std::vector<std::vector<cv::Point2f>>, cv::Size
>> _findPoints(const CalibrationSettings& settings, bool display);

CalibrationResult _calibrate(
    const CalibrationSettings& settings,
    std::vector<std::vector<cv::Point2f>>,
    cv::Size imageSize
);
};

#endif
