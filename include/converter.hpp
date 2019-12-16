#include<vector>

#include<opencv2/core.hpp>

namespace slam {

inline cv::Mat matFromVector(const std::vector<cv::Point2f>& p) {
    cv::Mat m(p.size(), 2, CV_32F);
    for (size_t i = 0; i < p.size(); i++) {
        const auto& t = p[i];
        m.at<float>(i, 0) = t.x;
        m.at<float>(i, 1) = t.y;
    }
    return m;
}

inline cv::Mat matFromVector(const std::vector<cv::Point3f>& p) {
    cv::Mat m(p.size(), 3, CV_32F);
    for (size_t i = 0; i < p.size(); i++) {
        const auto& t = p[i];
        m.at<float>(i, 0) = t.x;
        m.at<float>(i, 1) = t.y;
        m.at<float>(i, 2) = t.z;
    }
    return m;
}

inline std::vector<cv::Point3f> vectorFromMat(const cv::Mat& m) {
    std::vector<cv::Point3f> p;
    for (size_t i = 0; i < m.rows; i++)
        p.push_back(cv::Point3f(
            m.at<float>(i, 0), m.at<float>(i, 1), m.at<float>(i, 2)
        ));
    return p;
}

};
