#include<algorithm>
#include<iostream>

#include<opencv2/core/types.hpp>

#include"frame.hpp"

namespace slam {

bool Frame::initial = true;
float Frame::gridColsInv, Frame::gridRowsInv;

Frame::Frame(
    cv::Mat& image, const double& timestamp, Detector& detector
) : image(image), timestamp(timestamp), detector(detector) {
    if (initial) {
        gridColsInv = (
            static_cast<float>(GRID_COLS)
            / static_cast<float>(image.cols)
        );
        gridRowsInv = (
            static_cast<float>(GRID_ROWS)
            / static_cast<float>(image.rows)
        );
        initial = false;
    }

    /* Detect and extract keypoints and their descriptors */
    detector.detect(image, keypoints, descriptors);
    if (keypoints.empty()) return;

    _populateGrid();
    outliers = std::vector<bool>(keypoints.size(), false);
}

std::vector<size_t> Frame::getAreaFeatures(
    const float x, const float y, const float radius,
    const int minLevel, const int maxLevel
) const {
    std::vector<size_t> indices;

    int minCellX = std::max(
        0, static_cast<int>(floor((x - radius) * gridColsInv))
    );
    if (minCellX >= GRID_COLS) return indices;
    int minCellY = std::max(
        0, static_cast<int>(floor((y - radius) * Frame::gridColsInv))
    );
    if (minCellY >= GRID_ROWS) return indices;
    int maxCellX = std::min(
        GRID_COLS - 1, static_cast<int>(ceil((x + radius) * Frame::gridColsInv))
    );
    if (maxCellX < 0) return indices;
    int maxCellY = std::min(
        GRID_ROWS - 1, static_cast<int>(ceil((y + radius) * Frame::gridRowsInv))
    );
    if (maxCellY < 0) return indices;

    indices.reserve(keypoints.size());

    bool checkLevels = (minLevel != -1 && maxLevel != -1);
    bool sameLevel = (checkLevels && (minLevel == maxLevel));

    for (size_t i = minCellX; i < maxCellX; i++) {
        for (size_t j = minCellY; j < maxCellY; j++) {
            const std::vector<size_t>& cell = grid[i][j];
            if (cell.empty()) continue;

            for (size_t k = 0; k < cell.size(); k++) {
                const cv::KeyPoint& kp = keypoints[cell[k]];

                if (checkLevels && !sameLevel) {
                    if (kp.octave < minLevel || kp.octave > maxLevel)
                        continue;
                } else if (sameLevel) {
                    if (kp.octave != minLevel) continue;
                }

                if (abs(kp.pt.x - x) > radius || abs(kp.pt.y - y) > radius)
                    continue;

                indices.push_back(cell[k]);
            }
        }
    }

    return indices;
}

void Frame::_populateGrid() {
    unsigned int reservedCellSize = static_cast<unsigned int>(
        0.5f * keypoints.size() / static_cast<float>(GRID_ROWS * GRID_COLS)
    );
    for (unsigned int i = 0; i < GRID_COLS; i++)
        for (unsigned int j = 0; j < GRID_ROWS; j++)
            grid[i][j].reserve(reservedCellSize);

    unsigned int x, y;
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint& keypoint = keypoints[i];

        x = static_cast<unsigned int>(round(keypoint.pt.x * Frame::gridColsInv));
        y = static_cast<unsigned int>(round(keypoint.pt.y * Frame::gridRowsInv));

        grid[x][y].push_back(i);
    }
}

};
