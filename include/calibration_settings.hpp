#ifndef CALIBRATION_SETTINGS_H
#define CALIBRATION_SETTINGS_H

#include<string>
#include<vector>

#include<opencv2/core.hpp>

namespace slam {

class CalibrationSettings {
public:
    cv::Size boardSize;
    float squareSize;

    std::vector<std::string> images;

    double aspectRatio;
    bool useFisheye;
    bool fixPrincipalPoint;
    bool zeroTangentialDistortion;
    std::vector<bool> fixKs;

    int flag;

private:
    void _calculateFlag();
public:
    CalibrationSettings() = default;
    ~CalibrationSettings() = default;

    void read(const cv::FileNode& node);

    void write(cv::FileStorage& fs) const;
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

};

#endif
