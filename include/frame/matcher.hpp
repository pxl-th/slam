#ifndef MATCHER_H
#define MATCHER_H

#pragma warning(push, 0)
#include<vector>

#include<opencv2/features2d.hpp>
#pragma warning(pop)

#include"frame.hpp"

namespace slam {

class Matcher{
private:
    cv::Ptr<cv::BFMatcher> matcher;

public:
    Matcher(cv::Ptr<cv::BFMatcher> matcher);
    ~Matcher() = default;

    /**
     * Find matches between descriptors in given frames.
     *
     * @param frame1 Frame which will be used to find matches with `frame2`.
     * @param frame2 Frame which will be used to find mathces with `frame1`
     * @param maximumDistance Maximum distance in terms of Hamming distance.
     * @param areaSize
     *  If you want to find matches for keypoints in an area
     *  in terms of pixel distance between keypoints, then specify distance.
     *  If you want to find matches between all keypoints,
     *  no matter what pixel distace, specify `-1`.
     * @return matches Output vector which will contain matches.
     */
    std::vector<cv::DMatch> frameMatch(
        Frame& frame1, Frame& frame2,
        float maximumDistance, float areaSize = -1
    );

};

};

#endif
