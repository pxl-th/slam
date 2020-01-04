#ifndef MATCHER_H
#define MATCHER_H

#pragma warning(push, 0)
#include<vector>

#include<opencv2/features2d.hpp>
#pragma warning(pop)

#include"frame.hpp"
#include"map/keyframe.hpp"

namespace slam {

class Matcher{
private:
    cv::Ptr<cv::BFMatcher> matcher;
public:
    Matcher() : matcher(cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMING, true)) {}
    Matcher(cv::Ptr<cv::BFMatcher> matcher);
    /**
     * Find matches between descriptors in given frames.
     *
     * @param frame1 Frame which will be used to find matches with `frame2`.
     * @param frame2 Frame which will be used to find mathces with `frame1`
     * @param maximumDistance Maximum distance in terms of Hamming distance.
     * @param areaSize If you want to find matches for keypoints in an area
     * in terms of pixel distance between keypoints, then specify distance.
     * If you want to find matches between all keypoints,
     * no matter what pixel distace, specify `-1`.
     * @return matches Output vector which will contain matches.
     * Where `query` indices are for the `frame1`
     * and `train` indices --- `frame2`.
     */
    std::vector<cv::DMatch> frameMatch(
        std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2,
        float maximumDistance, float areaSize = -1, int maxLevel = 4
    );

    /**
     * Find matches between KeyFrames.
     *
     * First, mappoints from `fromKeyFrame` (fKF) are projected onto
     * `toKeyFrame` (tKF) image plane. \n
     * Then we find matches between descriptors in keyframes
     * and filter out those mathces whose respective mappoints lie outside
     * of the regions defined by projected mappoints and `areaSize`.
     *
     * @param fromKeyFrame Frame which will be used to find matches with `frame2`.
     * @param toKeyFrame Frame which will be used to find mathces with `frame1`
     * @param maximumDistance Maximum distance in terms of Hamming distance.
     * @param areaSize If you want to find matches for keypoints in an area
     * in terms of pixel distance between keypoints, then specify distance.
     * If you want to find matches between all keypoints,
     * no matter what pixel distace, specify `-1`.
     * @return matches Output vector which will contain matches.
     * Where `query` indices are for the `fromKeyFrame`
     * and `train` indices --- `toKeyFrame`.
     */
    std::vector<cv::DMatch> projectionMatch(
        std::shared_ptr<KeyFrame> fromKeyFrame,
        std::shared_ptr<KeyFrame> toKeyFrame,
        float maximumDistance, float areaSize = -1, int maxLevel = 4
    );
private:
    std::vector<cv::Point2f> _projectMapPoints(
        std::shared_ptr<KeyFrame> fromKeyFrame,
        std::shared_ptr<KeyFrame> toKeyFrame
    );
};

};

#endif
