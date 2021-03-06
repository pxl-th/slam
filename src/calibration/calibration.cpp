#pragma warning(push, 0)
#include<iostream>

#include<opencv2/calib3d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#pragma warning(pop)

#include"calibration/calibration.hpp"
#include"calibration/calibration_settings.hpp"

namespace slam {

Calibration::Calibration(
    const CalibrationSettings& settings, bool display, float resize
) {
    auto [imagePoints, imageSize] = _findPoints(settings, display, resize);
    _calibrate(settings, imagePoints, imageSize);
}

std::tuple<std::vector<std::vector<cv::Point2f>>, cv::Size>
Calibration::_findPoints(
    const CalibrationSettings& settings, bool display, float resize
) {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    cv::Size imageSize;
    float scale = 1.0f;

    for (const std::string& image_file : settings.images) {
        std::cout << "Processing image " << image_file << std::endl;

        cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
        if (image.data == nullptr) {
            std::cerr
                << "Could not open image " << image_file
                << ". Aborting." << std::endl;
            break;
        }
        if (resize > 0) scale = resize / image.cols;
        cv::resize(image, image, cv::Size(), scale, scale);
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
            cv::imshow("Calibration", image);
            cv::waitKey(0);
            cv::destroyWindow("Calibration");
        }
    }

    if (imagePoints.size() != settings.images.size()) {
        std::string msg =
            "Calibration failed to retrieve image points from all images. "
            "Try again with different images.";
        std::cerr << msg << std::endl;
        throw std::invalid_argument(msg);
    }

    std::cout << "Done processing images." << std::endl;
    return {imagePoints, imageSize};
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
    std::cout << "Calibrating..." << std::endl;

    cameraMatrix = cv::Mat::eye(3, 3, CV_32F);
    distortions = cv::Mat::zeros(settings.useFisheye ? 4 : 8, 1, CV_32F);

    if (settings.flag & cv::CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<float>(0, 0) = static_cast<float>(settings.aspectRatio);

    float gridWidth = settings.squareSize * (settings.boardSize.width - 1);
    auto corners = _calculateBorderCornerPosition(settings);
    corners[0][settings.boardSize.width - 1].x = corners[0][0].x + gridWidth;

    std::vector<cv::Point3f> newCorners = corners[0];
    corners.resize(points.size(), corners[0]);

    if (settings.useFisheye) {
        cv::Mat _rotations, _translations;
        calibrationError = cv::fisheye::calibrate(
            corners, points, imageSize, cameraMatrix, distortions,
            _rotations, _translations, settings.flag
        );

        rotations.reserve(_rotations.rows);
        translations.reserve(_translations.rows);
        for (int i = 0; i < corners.size(); i++) {
            rotations.push_back(_rotations.row(i));
            translations.push_back(_translations.row(i));
        }
    } else {
        calibrationError = cv::calibrateCameraRO(
            corners, points, imageSize, settings.boardSize.width - 1,
            cameraMatrix, distortions,
            rotations, translations, newCorners,
            settings.flag | cv::CALIB_USE_LU
        );
    }

    std::cout << "Reprojection error: " << calibrationError << std::endl;
}

void Calibration::read(const cv::FileNode& node) {
    node["calibrationError"] >> calibrationError;
    node["cameraMatrix"] >> cameraMatrix;
    node["distortions"] >> distortions;

    cv::FileNode rSeq = node["rotations"];
    for (
        cv::FileNodeIterator iter = rSeq.begin(), iterEnd = rSeq.end();
        iter != iterEnd;
        ++iter
    ) {
        cv::Mat tmp;
        (*iter) >> tmp;
        rotations.push_back(tmp);
    }

    cv::FileNode tSeq = node["translations"];
    for (
        cv::FileNodeIterator iter = tSeq.begin(), iterEnd = tSeq.end();
        iter != iterEnd;
        ++iter
    ) {
        cv::Mat tmp;
        (*iter) >> tmp;
        translations.push_back(tmp);
    }
}

void Calibration::write(cv::FileStorage& fs) const {
    fs
        << "{"
        << "calibrationError" << calibrationError
        << "cameraMatrix" << cameraMatrix
        << "distortions" << distortions;

    fs << "rotations" << "[";
    for (const auto& r : rotations) fs << r;
    fs << "]";

    fs << "translations" << "[";
    for (const auto& t : translations) fs << t;
    fs << "]";

    fs << "}";
}

};
