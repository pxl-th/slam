#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include<opencv2/core.hpp>

#include"detector.hpp"

/* TODO:
 * - getFeaturesInArea
 * */

namespace slam {

constexpr const unsigned int GRID_COLS = 64, GRID_ROWS = 48;

class Frame {
public:
    static bool initial;
    static float gridColsInv, gridRowsInv;

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

private:
    void _populateGrid();
};

};

#endif
