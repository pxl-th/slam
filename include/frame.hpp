#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include<opencv2/core.hpp>

#include"detector.hpp"

namespace slam {

constexpr const int GRID_COLS = 64, GRID_ROWS = 48;

class Frame {
public:
    static bool initial;
    static float gridColsInv, gridRowsInv; // cols = width

    Detector& detector;

    cv::Mat image;
    cv::Mat descriptors;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<bool> outliers;
    std::vector<std::size_t> grid[GRID_COLS][GRID_ROWS];

    double timestamp;

public:
    Frame() = default;
    Frame(cv::Mat& image, const double& timestamp, Detector& detector);
    ~Frame() = default;

    /* *
     * Select indices of keypoints in square window, with center at
     * `(x, y)` and length of the side `2 * radius`.
     * */
    std::vector<size_t> getAreaFeatures(
        const float x, const float y, const float radius,
        const int minLevel, const int maxLevel
    ) const;
private:
    /* *
     * Assign features ids to grid cells
     * to reduce computational complexity.
     * */
    void _populateGrid();
};

};

#endif
