#include<iostream>

#include<opencv2/calib3d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

#include"calibration.hpp"
#include"calibration_settings.hpp"

namespace slam {

Calibration::Calibration(const CalibrationSettings& settings, bool display) {
    auto points = _findPoints(settings, display);
    if (!points) return;
    auto [imagePoints, imageSize] = points.value();

    _calibrate(settings, imagePoints, imageSize);
}

std::optional<std::tuple<std::vector<std::vector<cv::Point2f>>, cv::Size>>
Calibration::_findPoints(const CalibrationSettings& settings, bool display) {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;

    for (const std::string& image_file : settings.images) {
        std::cout << "Processing image " << image_file << std::endl;

        cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
        if (image.data == nullptr) {
            std::cerr
                << "Could not open image " << image_file
                << ". Aborting." << std::endl;
            break;
        }
        imageSize = image.size();

        std::vector<cv::Point2f> points;
        int boardFlags = cv::CALIB_CB_ADAPTIVE_THRESH
            | cv::CALIB_CB_NORMALIZE_IMAGE;
        if (!settings.useFisheye)
            boardFlags |= cv::CALIB_CB_FAST_CHECK;
        bool found = cv::findChessboardCorners(
            image, settings.boardSize, points, boardFlags
        );
        if (!found) {
            std::cerr
                << "Failed to find chessboard corners on the image "
                << image_file << ". Aborting." << std::endl;
            break;
        }

        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(
            grayImage, points,
            cv::Size(11, 11), cv::Size(-1, -1),
            cv::TermCriteria(
                cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1
            )
        );

        imagePoints.push_back(points);

        if (display) {
            cv::drawChessboardCorners(
                image, settings.boardSize, cv::Mat(points), found
            );
            float scale = 1080.0f / image.cols;
            cv::resize(image, image, cv::Size(), scale, scale);
            cv::imshow("Calibration", image);
            cv::waitKey(0);
            cv::destroyWindow("Calibration");
        }
    }

    if (imagePoints.size() != settings.images.size()) {
        std::cerr
            << "Calibration failed to retrieve image points from all images. "
            << "Try again with different images."
            << std::endl;
        return {};
    }

    return std::tuple{imagePoints, imageSize};
}

std::vector<std::vector<cv::Point3f>>
Calibration::_calculateBorderCornerPosition(
    const CalibrationSettings& settings
) {
    std::vector<std::vector<cv::Point3f>> corners(1);

    for (int i = 0; i < settings.boardSize.height; i++) {
        for (int j = 0; j < settings.boardSize.width; j++) {
            corners[0].push_back(cv::Point3f(
                j * settings.squareSize, i * settings.squareSize, 0
            ));
        }
    }
    return corners;
}

void Calibration::_calibrate(
    const CalibrationSettings& settings,
    const std::vector<std::vector<cv::Point2f>>& points,
    const cv::Size& imageSize
) {
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    distortionCoefficients = cv::Mat::zeros(
        settings.useFisheye ? 4 : 8, 1, CV_64F
    );

    if (settings.flag & cv::CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = settings.aspectRatio;

    float gridWidth = settings.squareSize * (settings.boardSize.width - 1);
    auto corners = _calculateBorderCornerPosition(settings);
    corners[0][settings.boardSize.width - 1].x = corners[0][0].x + gridWidth;

    std::vector<cv::Point3f> newCorners = corners[0];
    corners.resize(points.size(), corners[0]);

    double reprojectionError;

    if (settings.useFisheye) {
        cv::Mat _rotations, _translations;
        reprojectionError = cv::fisheye::calibrate(
            corners, points, imageSize, cameraMatrix, distortionCoefficients,
            _rotations, _translations, settings.flag
        );

        rotations.reserve(_rotations.rows);
        translations.reserve(_translations.rows);
        for (int i = 0; i < corners.size(); i++) {
            rotations.push_back(_rotations.row(i));
            translations.push_back(_translations.row(i));
        }
    } else {
        reprojectionError = cv::calibrateCameraRO(
            corners, points, imageSize, -1,
            cameraMatrix, distortionCoefficients,
            rotations, translations, newCorners,
            settings.flag | cv::CALIB_USE_LU
        );
    }

    std::cout << "Reprojection error: " << reprojectionError << std::endl;
}

};
