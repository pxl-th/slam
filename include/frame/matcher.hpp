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
public:
    cv::Ptr<cv::BFMatcher> matcher;
public:
    Matcher() : matcher(cv::BFMatcher::create(cv::BFMatcher::BRUTEFORCE_HAMMING, true)) {}
    Matcher(cv::Ptr<cv::BFMatcher> matcher);
    /**
     * Find matches between descriptors in given frames.
     *
     * @param keyframe1 KeyFrame which will be used to find matches with keyframe2.
     * @param keyframe2 KeyFrame which will be used to find mathces with keyframe1.
     * @param maximumDistance Maximum distance in terms of Hamming distance.
     * @param areaSize If you want to find matches for keypoints in an area
     * in terms of pixel distance between keypoints, then specify distance.
     * If you want to find matches between all keypoints,
     * no matter what pixel distace, specify `-1`.
     * @param maxLevel KeyPoint matches whose `octave` is higher than
     * this value will be discarded.
     * @param withMappoints If `true`, finding matched will be done only
     * using KeyPoints associated with MapPoints in keyframe1.
     * This option is useful in tracking to speed up it.
     * If `false` then matching will be done using all KeyPoints.
     * @return matches Output vector which will contain matches.
     * Where `query` indices are for the `frame1`
     * and `train` indices --- `frame2`.
     */
    std::vector<cv::DMatch> frameMatch(
        const std::shared_ptr<KeyFrame>& keyframe1,
        const std::shared_ptr<KeyFrame>& keyframe2,
        const std::vector<int>& ids,
        float maximumDistance = 300, float areaSize = -1, int maxLevel = 4
    ) const;
    /**
     * Find matches among KeyPoints that have corresponding MapPoints.
     * This can be useful when performing tracking or sharing MapPoints
     * between KeyFrames as we only need to match against KeyPoints with
     * MapPoints.
     *
     * @keyframe1 KeyFrame from which to take KeyPoints with MapPoints to match.
     * Only MapPoints from this KeyFrame will be taken into account.
     * @keyframe2 KeyFrame which will be matched against `keyframe1`.
     *
     * For other parameters see Matcher.frameMatch method.
     */
    std::vector<cv::DMatch> mappointsFrameMatch(
        const std::shared_ptr<KeyFrame>& keyframe1,
        const std::shared_ptr<KeyFrame>& keyframe2,
        float maximumDistance = 300, float areaSize = -1, int maxLevel = 4
    ) const;
    /**
     * Find matches among KeyPoints that does not have corresponding MapPoints.
     * This can be useful when we need to find more matches for the KeyFrame
     * that already has some MapPoints attached to it.
     *
     * @keyframe1 KeyFrame for which to avoid KeyPoints with MapPoints.
     * Only MapPoints from this KeyFrame will be taken into account.
     * @keyframe2 KeyFrame which will be matched against `keyframe1`.
     *
     * For other parameters see Matcher.frameMatch method.
     */
    std::vector<cv::DMatch> inverseMappointsFrameMatch(
        const std::shared_ptr<KeyFrame>& keyframe1,
        const std::shared_ptr<KeyFrame>& keyframe2,
        float maximumDistance = 300, float areaSize = -1, int maxLevel = 4
    ) const;
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
        const std::shared_ptr<KeyFrame>& fromKeyFrame,
        const std::shared_ptr<KeyFrame>& toKeyFrame,
        float maximumDistance = 300, float areaSize = -1, int maxLevel = 4
    ) const;
private:
    static std::vector<cv::DMatch> _filterMatches(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<std::vector<cv::DMatch>>& rawMatches,
        float areaSize, int maxLevel
    );

    static std::vector<cv::Point2f> _projectMapPoints(
        const std::shared_ptr<KeyFrame>& fromKeyFrame,
        const std::shared_ptr<KeyFrame>& toKeyFrame
    );
};

};

#endif
