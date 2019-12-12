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

void test_orb() {
    cv::Mat image1 = cv::imread(
        "C:\\Users\\tonys\\Pictures\\r.png", cv::IMREAD_GRAYSCALE
    );
    cv::Mat image2 = cv::imread(
        "C:\\Users\\tonys\\Pictures\\r3.png", cv::IMREAD_GRAYSCALE
    );

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        100, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE
    );
    slam::Detector detector(orb);

    cv::Mat descriptors1, descriptors2;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    detector.detect(image1, keypoints1, descriptors1);
    detector.detect(image2, keypoints2, descriptors2);

    slam::Matcher matcher(cv::DescriptorMatcher::create(
        cv::DescriptorMatcher::BRUTEFORCE_HAMMING
    ));
    std::vector<std::vector<cv::DMatch>> matches;

    matcher.knnMatch(descriptors1, descriptors2, matches, 2);
    std::cout << descriptors1.size() << std::endl;
    std::cout << descriptors2.size() << std::endl;
    std::cout << matches.size() << std::endl;

    const float ratio_thresh = 0.7f;
    std::vector<cv::DMatch> good_matches;
    /* for (size_t i = 0; i < matches.size(); i++) { */
    /*     good_matches.push_back(matches[i][0]); */
    /* } */
    for (size_t i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }

    cv::Mat img_matches;
    cv::drawMatches(
        image1, keypoints1, image2, keypoints2,
        good_matches, img_matches,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    cv::imshow("test", img_matches);
    cv::waitKey(0);
    cv::destroyWindow("test");
}

void test_frame() {
    cv::Mat image = cv::imread(
        "C:\\Users\\tonys\\Pictures\\r.png", cv::IMREAD_GRAYSCALE
    );
    slam::Detector detector(cv::ORB::create(
        1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE
    ));
    slam::Frame frame(image, 0, detector);

    auto ids = frame.getAreaFeatures(500, 500, 100, 1, 4);
    std::cout << ids.size() << std::endl;
}

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

    slam::Calibration calibration(settings, true, 1080);
    slam::save(calibration, outputFile, "Calibration");
}

int main() {
    /* test_orb(); */
    /* test_frame(); */
    test_settings();
}
