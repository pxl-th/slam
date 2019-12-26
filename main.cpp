#pragma warning(push, 0)
#include<iostream>
#include<string>

#include<opencv2/calib3d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/viz.hpp>
#pragma warning(pop)

#include"include/calibration/calibration.hpp"
#include"include/calibration/calibration_settings.hpp"
/* #include"include/frame/detector.hpp" */
/* #include"include/frame/frame.hpp" */
/* #include"include/tracking/initializer.hpp" */
/* #include"include/map/keyframe.hpp" */
/* #include"include/frame/matcher.hpp" */
/* #include"include/map/mappoint.hpp" */
/* #include"include/map/map.hpp" */
#include"include/loader.hpp"
#include"tracking/tracker.hpp"

slam::Calibration getCalibration(bool compute = false) {
    std::string outputFile(
        "C:\\Users\\tonys\\projects\\cpp\\slam\\data\\calibration.yaml"
    );
    if (!compute)
        return slam::load<slam::Calibration>(outputFile, "Calibration");

    std::string settingsFile(
        "C:\\Users\\tonys\\projects\\cpp\\slam\\data\\settings.yaml"
    );
    slam::CalibrationSettings settings = slam::load<slam::CalibrationSettings>(
        settingsFile, "CalibrationSettings"
    );
    slam::Calibration calibration(settings, false, 1080);
    slam::save(calibration, outputFile, "Calibration");
    return calibration;
}

int main() {
    auto calibration = getCalibration();
    auto initImage = std::make_shared<cv::Mat>(cv::imread(
        "C:\\Users\\tonys\\Downloads\\1.jpg", cv::IMREAD_GRAYSCALE
    ));
    auto currentImage = std::make_shared<cv::Mat>(cv::imread(
        "C:\\Users\\tonys\\Downloads\\2.jpg", cv::IMREAD_GRAYSCALE
    ));
    auto detector = std::shared_ptr<slam::Detector>(new slam::Detector(
        cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE)
    ));

    slam::Tracker tracker(calibration, detector);
    tracker.track(initImage);
    tracker.track(currentImage);

    /* Visualization */
    std::vector<cv::Point3f> adjustedPoints;
    for (const auto& p : tracker.map->getMappoints())
        adjustedPoints.push_back(p->getWorldPos());

    cv::viz::Viz3d window("slam");
    cv::viz::WCloud cloud(adjustedPoints, cv::viz::Color::red());
    cv::viz::WCoordinateSystem coordinateSystem;
    while (!window.wasStopped()) {
        window.showWidget("CS", coordinateSystem);
        window.showWidget("cloud", cloud);
        window.spinOnce(1, true);
    }
}

/* int main() { */
/*     auto mz = std::make_shared<cv::Mat>(cv::Mat(2, 1, CV_32F)); */
/*     mz->at<float>(0) = 2.5f; */

/*     std::cout << *mz << std::endl; */
/*     cv::Mat m(*mz); */

/*     std::cout << m << std::endl; */
/*     m.at<float>(1) = 5.0f; */
/*     std::cout << m << std::endl; */

/*     std::cout << *mz << std::endl; */
/*     m.copyTo(*mz); */
/*     std::cout << *mz << std::endl; */
/* } */
