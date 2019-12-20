#include<iostream>

#include<opencv2/calib3d.hpp>

#include"calibration/calibration_settings.hpp"

namespace slam {

void CalibrationSettings::_calculateFlag() {
    if (useFisheye) {
        int fixKsValues[4] = {
            cv::fisheye::CALIB_FIX_K1, cv::fisheye::CALIB_FIX_K2,
            cv::fisheye::CALIB_FIX_K3, cv::fisheye::CALIB_FIX_K4
        };
        flag = cv::fisheye::CALIB_FIX_SKEW;
        flag |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;

        if (fixPrincipalPoint) flag |= cv::fisheye::CALIB_FIX_PRINCIPAL_POINT;
        for (int i = 0; i < 4; i++) if (fixKs[i]) flag |= fixKsValues[i];

    } else {
        int fixKsValues[5] = {
            cv::CALIB_FIX_K1, cv::CALIB_FIX_K2, cv::CALIB_FIX_K3,
            cv::CALIB_FIX_K4, cv::CALIB_FIX_K5
        };
        flag = 0;
    }
}

void CalibrationSettings::read(const cv::FileNode& node) {
    node["boardWidth"] >> boardSize.width;
    node["boardHeight"] >> boardSize.height;
    node["squareSize"] >> squareSize;

    node["aspectRatio"] >> aspectRatio;
    node["useFisheye"] >> useFisheye;
    node["fixPrincipalPoint"] >> fixPrincipalPoint;
    node["zeroTangentialDistortion"] >> zeroTangentialDistortion;

    cv::FileNode kSeq = node["fixKs"];
    for (
        cv::FileNodeIterator iter = kSeq.begin(), iterEnd = kSeq.end();
        iter != iterEnd;
        ++iter
    )
        fixKs.push_back(static_cast<int>(*iter));

    cv::FileNode iSeq = node["images"];
    for (
        cv::FileNodeIterator iter = iSeq.begin(), iterEnd = iSeq.end();
        iter != iterEnd;
        ++iter
    )
        images.push_back(static_cast<std::string>(*iter));

    _calculateFlag();
}

void CalibrationSettings::write(cv::FileStorage& fs) const {
    fs
        << "{"
        << "boardWidth" << boardSize.width
        << "boardHeight" << boardSize.height
        << "squareSize" << squareSize
        << "aspectRatio" << aspectRatio
        << "useFisheye" << useFisheye
        << "fixPrincipalPoint" << fixPrincipalPoint
        << "zeroTangentialDistortion" << zeroTangentialDistortion;

    fs << "fixKs" << "[";
    for (const auto& i : fixKs) fs << i;
    fs << "]";

    fs << "images" << "[";
    for (const auto& i : images) fs << i;
    fs << "]";

    fs << "}";
}

};
