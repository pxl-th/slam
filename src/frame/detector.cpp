#include<iostream>
#include"frame/detector.hpp"

slam::Detector::Detector(cv::Ptr<cv::ORB> detector) : detector(detector) {}

void slam::Detector::detect(
    std::shared_ptr<cv::Mat> image,
    std::vector<cv::KeyPoint>& keypoints,
    std::shared_ptr<cv::Mat> descriptors
) {
    detector->detectAndCompute(*image, cv::noArray(), keypoints, *descriptors);
}
