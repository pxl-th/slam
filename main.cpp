#include<iostream>
#include<string>

#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>

#include"include/calibration.hpp"
#include"include/calibration_settings.hpp"
#include"include/detector.hpp"
#include"include/frame.hpp"
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
    slam::Detector detector(
        cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE)
    );
    slam::Matcher matcher(cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMING, true));

    slam::Frame frame1(
        cv::imread("C:\\Users\\tonys\\Pictures\\r.png", cv::IMREAD_GRAYSCALE),
        0, detector, calibration.cameraMatrix, calibration.distortions
    );
    slam::Frame frame2(
        cv::imread("C:\\Users\\tonys\\Pictures\\r.png", cv::IMREAD_GRAYSCALE),
        0, detector, calibration.cameraMatrix, calibration.distortions
    );

    std::vector<cv::DMatch> matches;

    matcher.frameMatch(frame1, frame2, matches, 50, 100);
}

int main() {
    test_settings();
}
