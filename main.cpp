#include<iostream>
#include<string>

#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/viz.hpp>

#include"include/calibration.hpp"
#include"include/calibration_settings.hpp"
#include"include/detector.hpp"
#include"include/frame.hpp"
#include"include/initializer.hpp"
#include"include/matcher.hpp"
#include"include/loader.hpp"

void test_settings() {
    std::string settingsFile(
        "C:\\Users\\tonys\\projects\\cpp\\slam\\data\\settings.yaml"
    );
    std::string outputFile(
        "C:\\Users\\tonys\\projects\\cpp\\slam\\data\\calibration.yaml"
    );
    slam::CalibrationSettings settings = slam::load<slam::CalibrationSettings>(
        settingsFile, "CalibrationSettings"
    );
    /* slam::Calibration calibration(settings, false, 1080); */
    /* slam::save(calibration, outputFile, "Calibration"); */
    auto calibration = slam::load<slam::Calibration>(outputFile, "Calibration");
    slam::Detector detector(cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE));
    slam::Matcher matcher(cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMING, true));

    slam::Frame frame1(
        cv::imread("C:\\Users\\tonys\\Downloads\\1.jpg", cv::IMREAD_GRAYSCALE),
        0, detector, calibration.cameraMatrix, calibration.distortions
    );
    slam::Frame frame2(
        cv::imread("C:\\Users\\tonys\\Downloads\\2.jpg", cv::IMREAD_GRAYSCALE),
        0, detector, calibration.cameraMatrix, calibration.distortions
    );

    std::vector<cv::DMatch> matches;

    matcher.frameMatch(frame1, frame2, matches, 250, 50);
    std::cout << "Frame matches " << matches.size() << std::endl;
    slam::Initializer initializer(frame1);
    auto [rotation, translation, inliersMask, reconstructedPoints] = initializer.initialize(frame2, matches);
    std::cout << "Reconstructed points " << reconstructedPoints.size() << std::endl;

    cv::viz::Viz3d window("slam");
    cv::viz::WCloud cloud(reconstructedPoints, cv::viz::Color::white());
    cv::viz::WCoordinateSystem coordinateSystem;

    while (!window.wasStopped()) {
        window.showWidget("CS", coordinateSystem);
        window.showWidget("cloud", cloud);
        window.spinOnce(1, true);
    }
}

int main() {
    test_settings();
}
