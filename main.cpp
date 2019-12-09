#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include"include/orb.hpp"

void test_orb() {
    cv::Mat image = cv::imread(
        "C:\\Users\\tonys\\Pictures\\r.png", cv::IMREAD_GRAYSCALE
    );

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE
    );
    slam::FeatureDetector detector(orb);

    cv::Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(image, keypoints, descriptors);

    cv::drawKeypoints(image, keypoints, image);
    cv::imshow("test", image);
    cv::waitKey(0);
    cv::destroyWindow("test");
}

int main() {
    test_orb();
}
