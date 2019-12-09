#ifndef MATCHER_H
#define MATCHER_H

#include<vector>

#include<opencv2/features2d.hpp>

namespace slam {

class Matcher{
private:
    cv::Ptr<cv::DescriptorMatcher> matcher;

public:
    Matcher(cv::Ptr<cv::DescriptorMatcher> matcher);
    ~Matcher() = default;

    void knnMatch(
        const cv::Mat& descriptors1, const cv::Mat& descriptors2,
        std::vector<std::vector<cv::DMatch>>& matches,
        const int k = 1
    );
};

};

#endif
