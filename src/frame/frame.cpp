#include<algorithm>
#include<iostream>

#include<opencv2/core.hpp>
#include<opencv2/core/types.hpp>
#include<opencv2/calib3d.hpp>

#include"frame/frame.hpp"

namespace slam {

bool Frame::initial = true;
float Frame::gridColsInv, Frame::gridRowsInv;
int Frame::imageBounds[4];

Frame::Frame(
    cv::Mat& image, const double& timestamp, Detector& detector,
    cv::Mat& cameraMatrix, cv::Mat& distortions
) : image(image), timestamp(timestamp), detector(detector),
    cameraMatrix(cameraMatrix.clone()), distortions(distortions.clone()) {
    if (initial) {
        _undistortImageBounds();

        gridColsInv = (
            static_cast<float>(GRID_COLS)
            / static_cast<float>(imageBounds[1] - imageBounds[0])
        );
        gridRowsInv = (
            static_cast<float>(GRID_ROWS)
            / static_cast<float>(imageBounds[3] - imageBounds[2])
        );
        initial = false;
    }

    /* Detect and extract keypoints and their descriptors */
    detector.detect(image, keypoints, descriptors);
    if (keypoints.empty()) return;
    _undistortKeyPoints();

    _populateGrid();
    outliers = std::vector<bool>(keypoints.size(), false);
}

std::vector<int> Frame::getAreaFeatures(
    float x, float y, float side, int minLevel, int maxLevel
) const {
    std::vector<int> indices;

    int minCellX = std::max(
        0, static_cast<int>(floor((x - side) * gridColsInv))
    );
    if (minCellX >= GRID_COLS) return indices;
    int minCellY = std::max(
        0, static_cast<int>(floor((y - side) * Frame::gridColsInv))
    );
    if (minCellY >= GRID_ROWS) return indices;
    int maxCellX = std::min(
        GRID_COLS - 1, static_cast<int>(ceil((x + side) * Frame::gridColsInv))
    );
    if (maxCellX < 0) return indices;
    int maxCellY = std::min(
        GRID_ROWS - 1, static_cast<int>(ceil((y + side) * Frame::gridRowsInv))
    );
    if (maxCellY < 0) return indices;

    indices.reserve(keypoints.size());

    bool checkLevels = (minLevel != -1 && maxLevel != -1);
    bool sameLevel = (checkLevels && (minLevel == maxLevel));

    for (int i = minCellX; i < maxCellX; i++) {
        for (int j = minCellY; j < maxCellY; j++) {
            const std::vector<int>& cell = grid[i][j];
            if (cell.empty()) continue;

            for (int k = 0; k < static_cast<int>(cell.size()); k++) {
                const cv::KeyPoint& kp = keypoints[cell[k]];

                if (checkLevels && !sameLevel) {
                    if (kp.octave < minLevel || kp.octave > maxLevel)
                        continue;
                } else if (sameLevel) {
                    if (kp.octave != minLevel) continue;
                }

                if (abs(kp.pt.x - x) > side || abs(kp.pt.y - y) > side)
                    continue;

                indices.push_back(cell[k]);
            }
        }
    }

    return indices;
}

void Frame::_populateGrid() {
    size_t x, y;
    for (size_t i = 0; i < undistortedKeypoints.size(); i++) {
        cv::KeyPoint& keypoint = undistortedKeypoints[i];

        x = static_cast<size_t>(round(
            (keypoint.pt.x - imageBounds[0]) * Frame::gridColsInv
        ));
        y = static_cast<size_t>(round(
            (keypoint.pt.y - imageBounds[2]) * Frame::gridRowsInv
        ));

        if (x < 0 || x > GRID_COLS || y < 0 || y > GRID_ROWS)
            continue;

        grid[x][y].push_back(static_cast<int>(i));
    }
}

void Frame::_undistortKeyPoints() {
    // If no distortion, then nothing to undistort.
    if (distortions.at<float>(0) == 0.0f) {
        undistortedKeypoints = keypoints;
        return;
    }

    // Convert keypoints to matrix representation.
    cv::Mat undistorted(static_cast<int>(keypoints.size()), 2, CV_64F);
    for (int i = 0; i < keypoints.size(); i++) {
        undistorted.at<float>(i, 0) = keypoints[i].pt.x;
        undistorted.at<float>(i, 1) = keypoints[i].pt.y;
    }

    undistorted.reshape(2);
    cv::undistortPoints(
        undistorted, undistorted, cameraMatrix,
        distortions, cv::Mat(), cameraMatrix
    );
    undistorted.reshape(1);

    // Convert undistorted keypoints from matrix representation to list.
    undistortedKeypoints.resize(keypoints.size());
    for(int i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint p = keypoints[i];
        p.pt.x = undistorted.at<float>(i, 0);
        p.pt.y = undistorted.at<float>(i, 1);
        undistortedKeypoints[i] = p;
    }
}

void Frame::_undistortImageBounds() {
    // If no distortion, then nothing to undistort.
    if (distortions.at<float>(0) == 0.0) {
        imageBounds[0] = 0; imageBounds[1] = image.cols;
        imageBounds[2] = 0; imageBounds[3] = image.rows;
        return;
    }

    cv::Mat bounds = cv::Mat::zeros(4, 2, CV_32F);
    bounds.at<float>(1, 0) = static_cast<float>(image.cols);
    bounds.at<float>(2, 1) = static_cast<float>(image.rows);
    bounds.at<float>(3, 0) = static_cast<float>(image.cols);
    bounds.at<float>(3, 1) = static_cast<float>(image.rows);

    bounds.reshape(2);
    cv::undistortPoints(
        bounds, bounds, cameraMatrix, distortions, cv::Mat(), cameraMatrix
    );
    bounds.reshape(1);

    // Convert from matrix view.
    imageBounds[0] = static_cast<int>(std::min(
        floor(bounds.at<double>(0, 0)), floor(bounds.at<double>(2, 0))
    ));
    imageBounds[1] = static_cast<int>(std::max(
        ceil(bounds.at<double>(1, 0)), ceil(bounds.at<double>(3, 0))
    ));
    imageBounds[2] = static_cast<int>(std::min(
        floor(bounds.at<double>(0, 1)), floor(bounds.at<double>(1, 1))
    ));
    imageBounds[3] = static_cast<int>(std::max(
        ceil(bounds.at<double>(2, 1)), ceil(bounds.at<double>(3, 1))
    ));
}

};
