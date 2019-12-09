#include<iostream>

#include"orb.hpp"


slam::FeatureDetector::FeatureDetector(cv::Ptr<cv::FeatureDetector> detector)
    : detector(detector) {}


void slam::FeatureDetector::detect(
    const cv::InputArray& image,
    std::vector<cv::KeyPoint>& keypoints, cv::OutputArray& descriptors
) {
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}
