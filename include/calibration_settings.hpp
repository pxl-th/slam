#ifndef CALIBRATION_SETTINGS_H
#define CALIBRATION_SETTINGS_H

#include<string>
#include<vector>
#include<optional>

#include<opencv2/core.hpp>

namespace slam {

/**
 * Class that contains settings used in camera calibration process.
 * Calibration procedure supports only
 * [chessboard](https://docs.opencv.org/master/d9/d0c/
 * group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a) pattern.
 */
class CalibrationSettings {
public:
    /**
     * Number of corners on the chessboard.
     */
    cv::Size boardSize;
    /**
     * Size of the side of the square on the chessboard.
     * Could be in millimeters, pixels, etc.
     */
    float squareSize;
    /**
     * List of image path's to use in calibration.
     * Each image should contain a chessboard pattern in it.
     */
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

    /**
     * Deserialize class using given `cv::FileNode` in-place.
     *
     * @param node `cv::FileNode` that contains info for deserialization.
     */
    void read(const cv::FileNode& node);
    /**
     * Serialize instance of a class.
     *
     * @param fs `cv::FileStorage` which will handle serialization.
     */
    void write(cv::FileStorage& fs) const;
};

};

#endif
