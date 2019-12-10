#include<iostream>

#include<opencv2/calib3d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include"calibration.hpp"

namespace slam {

CalibrationSettings::CalibrationSettings()
    : boardSize(0, 0), squareSize(0) {}

void CalibrationSettings::read(const cv::FileNode& node) {
    node["boardWidth"] >> boardSize.width;
    node["boardHeight"] >> boardSize.height;
    node["squareSize"] >> squareSize;

    cv::FileNode seq = node["images"];
    for (
        cv::FileNodeIterator iter = seq.begin(), iterEnd = seq.end();
        iter != iterEnd;
        ++iter
    )
        images.push_back(static_cast<std::string>(*iter));
}

void CalibrationSettings::write(cv::FileStorage& fs) const {
    fs
        << "{"
        << "boardWidth" << boardSize.width
        << "boardHeight" << boardSize.height
        << "squareSize" << squareSize
        << "}";
}

static void write(
    cv::FileStorage& fs,
    const std::string&,
    const CalibrationSettings& settings
) {
    settings.write(fs);
}

void read(
    const cv::FileNode& node,
    CalibrationSettings& settings,
    const CalibrationSettings& defaultSettings
) {
    if(node.empty())
        settings = defaultSettings;
    else
        settings.read(node);
}

CalibrationSettings loadCalibrationSettings(const std::string& settingsFile) {
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr
            << "Could not open configuration file"
            << settingsFile << std::endl;
        return {};
    }

    CalibrationSettings settings;
    fs["CalibrationSettings"] >> settings;
    fs.release();

    return settings;
}

void calibrateCamera(
    const CalibrationSettings& settings, const std::string& outputFile,
    bool display
) {
    auto points = _findPoints(settings, display);
    if (!points) return;
    auto [imagePoints, imageSize] = points.value();
}

std::optional<std::tuple<
    std::vector<std::vector<cv::Point2f>>, cv::Size
>> _findPoints(const CalibrationSettings& settings, bool display) {
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
        bool found = cv::findChessboardCorners(
            image, settings.boardSize, points,
            cv::CALIB_CB_ADAPTIVE_THRESH
            | cv::CALIB_CB_FAST_CHECK
            | cv::CALIB_CB_NORMALIZE_IMAGE
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

CalibrationResult _calibrate(
    const CalibrationSettings& settings,
    std::vector<std::vector<cv::Point2f>>,
    cv::Size imageSize
) {
    return CalibrationResult();
}

};
