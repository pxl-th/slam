#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include<opencv2/core.hpp>

#include"detector.hpp"

namespace slam {

/**
 * Sizes of the grid used to split keypoints
 * to reduce computational complexity.
 */
constexpr const int GRID_COLS = 64, GRID_ROWS = 48;

/**
 * Frame class that contains functions for holding and retrieving
 * info (e.g. keypoints) from the frame.
 */
class Frame {
public:
    /**
     * `true` if this is first ever initialized frame.
     * In this case it computes image bounds,
     * and inverted number of columns and rows in the grid.
     * After that it is always `false`.
     */
    static bool initial;
    static float gridColsInv, gridRowsInv;
    /**
     * Undistorted image bounds in `[minX, maxX, minY, maxY]` format.
     */
    static int imageBounds[4];

    Detector& detector;

    cv::Mat image;
    cv::Mat descriptors;

    std::vector<cv::KeyPoint> keypoints, undistortedKeypoints;
    std::vector<bool> outliers;
    std::vector<std::size_t> grid[GRID_COLS][GRID_ROWS];

    cv::Mat cameraMatrix, distortions;

    double timestamp;

public:
    Frame() = default;
    /**
     * Create frame from given image.
     *
     * @param image Image of the current frame.
     * @param timestamp Timestamp of the image.
     * @param detector Keypoints detector for feature extraction.
     * @param cameraMatrix Camera matrix calculated using Calibration.
     * @param distortions Distortions coefficients calculated using Calibration.
     */
    Frame(
        cv::Mat& image, const double& timestamp, Detector& detector,
        cv::Mat& cameraMatrix, cv::Mat& distortions
    );
    ~Frame() = default;

    /**
     * Select indices of keypoints in square window area.
     *
     * @param x x-coordinate of the center of the square from which to extract
     * keypoints.
     * @param y y-coordinate of the center of the square from which to extract
     * keypoints.
     * @param side Size of the side of the square.
     * @param minLevel Minimal [level](https://docs.opencv.org/4.1.2/d2/d29/
     * classcv_1_1KeyPoint.html#aee152750aa98ea54a48196a937197095)
     * of the Detector's pyramid, from which to extract features.
     * @param maxLevel Maximum [level](https://docs.opencv.org/4.1.2/d2/d29/
     * classcv_1_1KeyPoint.html#aee152750aa98ea54a48196a937197095)
     * of the Detector's pyramid, from which to extract features.
     * In order to extract from any level, specify both
     * `minLevel` and `maxLevel` as `-1`.
     */
    std::vector<size_t> getAreaFeatures(
        const float x, const float y, const float side,
        const int minLevel = -1, const int maxLevel = -1
    ) const;
private:
    /**
     * Assign features ids to grid cells
     * to reduce computational complexity.
     */
    void _populateGrid();

    void _undistortKeyPoints();

    /**
     * Calculate undistorted image bounds for the undistorted image.
     */
    void _undistortImageBounds();
};

};

#endif
