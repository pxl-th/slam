#include "matcher.hpp"

slam::Matcher::Matcher(cv::Ptr<cv::DescriptorMatcher> matcher)
    : matcher(matcher) {}

void slam::Matcher::knnMatch(
    const cv::Mat& descriptors1, const cv::Mat& descriptors2,
    std::vector<std::vector<cv::DMatch>>& matches,
    const int k
) {
    matcher->knnMatch(descriptors1, descriptors2, matches, k);
}
