#pragma warning(push, 0)
#include<vector>

#include<opencv2/core.hpp>
#include<Eigen/Core>
#include<g2o/types/sim3/types_seven_dof_expmap.h>
#pragma warning(pop)

namespace slam {

inline cv::Mat matFromPoint3f(const cv::Point3f& p) {
    cv::Mat m(1, 3, CV_32F);
    m.at<float>(0) = p.x;
    m.at<float>(1) = p.y;
    m.at<float>(2) = p.z;
    return m;
}

inline cv::Mat matFromVector(const std::vector<cv::Point2f>& p) {
    int size = static_cast<int>(p.size());
    cv::Mat m(size, 2, CV_32F);
    for (int i = 0; i < size; i++) {
        const auto& t = p[i];
        m.at<float>(i, 0) = t.x;
        m.at<float>(i, 1) = t.y;
    }
    return m;
}

inline cv::Mat matFromVector(const std::vector<cv::Point3f>& p) {
    int size = static_cast<int>(p.size());
    cv::Mat m(size, 3, CV_32F);
    for (int i = 0; i < size; i++) {
        const auto& t = p[i];
        m.at<float>(i, 0) = t.x;
        m.at<float>(i, 1) = t.y;
        m.at<float>(i, 2) = t.z;
    }
    return m;
}

inline cv::Mat toHomogeneous(const cv::Mat& m) {
    cv::Mat r = cv::Mat::ones(m.rows, m.cols + 1, CV_32F);
    m.copyTo(r.rowRange(0, m.rows).colRange(0, m.cols));
    return r;
}

inline cv::Mat fromHomogeneous(const cv::Mat& m) {
    cv::Mat r(m.rows, m.cols - 1, CV_32F);
    m.rowRange(0, m.rows).colRange(0, m.cols - 1).copyTo(r);
    return r;
}

inline std::vector<cv::Point3f> vectorFromMat(const cv::Mat& m) {
    std::vector<cv::Point3f> p;
    for (int i = 0; i < m.rows; i++)
        p.push_back(cv::Point3f(
            m.at<float>(i, 0), m.at<float>(i, 1), m.at<float>(i, 2)
        ));
    return p;
}

/**
 * Convert `4x4` transformation matrix `cv::Mat`
 * to `g2o::SE3Quat` quaternion.
 *
 * @param m `cv::Mat` transformation matrix to convert.
 * Must have `4x4` shape.
 * @return `g2o::SE3Quat` quaternion calculated from `m`.
 */
inline g2o::SE3Quat matToSE3Quat(const cv::Mat& m) {
    Eigen::Matrix<double,3,3> R;
    R << m.at<float>(0,0), m.at<float>(0,1), m.at<float>(0,2),
         m.at<float>(1,0), m.at<float>(1,1), m.at<float>(1,2),
         m.at<float>(2,0), m.at<float>(2,1), m.at<float>(2,2);
    Eigen::Matrix<double,3,1> t(
        m.at<float>(0,3), m.at<float>(1,3), m.at<float>(2,3)
    );
    return g2o::SE3Quat(R, t);
}

inline cv::Mat se3QuatToMat(const g2o::SE3Quat& q) {
    Eigen::Matrix<double, 4, 4> h = q.to_homogeneous_matrix();
    cv::Mat m(4, 4, CV_32F);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m.at<float>(i, j) = static_cast<float>(h(i, j));
    return m.clone();
}

inline Eigen::Matrix<double, 3, 1> pointToVec3d(const cv::Point3f& p) {
    Eigen::Matrix<double, 3, 1> v;
    v << p.x, p.y, p.z;
    return v;
}

inline cv::Point3f vec3dToPoint3f(const Eigen::Matrix<double, 3, 1>& m) {
    cv::Point3f p;
    p.x = static_cast<float>(m[0]);
    p.y = static_cast<float>(m[1]);
    p.z = static_cast<float>(m[2]);
    return p;
}

};
