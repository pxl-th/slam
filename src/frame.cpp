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

/* Assign features ids to grid cells to reduce computational complexity.
 * */
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

        x = static_cast<unsigned int>(round(keypoint.pt.x * gridColsInv));
        y = static_cast<unsigned int>(round(keypoint.pt.y * gridRowsInv));

        grid[x][y].push_back(i);
    }
}

};
