#include"frame/detector.hpp"

slam::Detector::Detector(cv::Ptr<cv::ORB> detector) : detector(detector) {}

void slam::Detector::detect(
    cv::InputArray& image,
    std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors
) {
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}
