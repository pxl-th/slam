#ifndef MATCHER_H
#define MATCHER_H

#include<vector>

#include<opencv2/features2d.hpp>

#include"frame.hpp"

namespace slam {

class Matcher{
private:
    float distanceRatio;
    cv::Ptr<cv::BFMatcher> matcher;

public:
    Matcher(cv::Ptr<cv::BFMatcher> matcher);
    ~Matcher() = default;

    /**
     * Find matches between given descriptors.
     *
     * @param[in] frame1 Frame which will be used to find matches with `frame2`.
     * @param[in] frame2 Frame which will be used to find mathces with `frame1`
     * @param[out] matches Output vector which will contain matches.
     * @param maximumDistance Maximum distance in terms of Hamming distance.
     * @param areaSize
     *  If you want to find matches for keypoints in an area
     *  in terms of pixel distance between keypoints, then specify distance.
     *  If you want to find matches between all keypoints,
     *  no matter what pixel distace, specify `-1`.
     */
    void frameMatch(
        const Frame& frame1, const Frame& frame2,
        std::vector<cv::DMatch>& matches,
        float maximumDistance, float areaSize = -1
    );

};

};

#endif
