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
#include"include/loader.hpp"
#include"tracking/tracker.hpp"
#include"converter.hpp"

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

std::tuple<cv::viz::WCameraPosition, cv::Affine3d>
cameraFromKeyFrame(std::shared_ptr<slam::KeyFrame> keyframe) {
    cv::Matx44f projection = keyframe->getPose();
    cv::viz::Camera camera(projection, cv::Size(1280, 960));
    cv::Affine3d cameraPosition = cv::viz::makeCameraPose(
        keyframe->getCameraCenter(),
        cv::Vec3f(0.0f, 0.0f, 1.0f),
        cv::Vec3f(0.0f, 1.0f, 1.0f)
    );
    return {
        cv::viz::WCameraPosition(camera.getFov()), cameraPosition
    };
}

int main() {
    auto calibration = getCalibration();
    auto initImage = std::make_shared<cv::Mat>(cv::imread(
        "C:\\Users\\tonys\\Downloads\\1.jpg", cv::IMREAD_GRAYSCALE
    ));
    auto currentImage = std::make_shared<cv::Mat>(cv::imread(
        "C:\\Users\\tonys\\Downloads\\2.jpg", cv::IMREAD_GRAYSCALE
    ));
    auto nextImage = std::make_shared<cv::Mat>(cv::imread(
        "C:\\Users\\tonys\\Downloads\\3.jpg", cv::IMREAD_GRAYSCALE
    ));
    auto detector = std::shared_ptr<slam::Detector>(new slam::Detector(
        cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE)
    ));

    slam::Tracker tracker(calibration, detector);
    tracker.track(initImage);
    tracker.track(currentImage);
    /* tracker.track(nextImage); */

    auto targetKeyFrame = tracker.map->getKeyframes()[1];
    auto [cameraWidget, cameraPosition] = cameraFromKeyFrame(targetKeyFrame);

    /* Visualization */
    std::vector<cv::Point3f> adjustedPoints;
    for (const auto& p : tracker.map->getMappoints())
        adjustedPoints.push_back(p->getWorldPos());
    std::cout << "example" << std::endl;

    cv::Mat keyframePose = targetKeyFrame->getPose();
    auto pointsHomo = slam::toHomogeneous(slam::matFromVector(adjustedPoints));
    adjustedPoints = slam::vectorFromMat(slam::fromHomogeneous(
        pointsHomo * keyframePose.t()
    ));

    cv::viz::Viz3d window("slam");
    cv::viz::WCloud cloud(adjustedPoints, cv::viz::Color::red());
    cv::viz::WCoordinateSystem coordinateSystem;
    while (!window.wasStopped()) {
        window.showWidget("CW", cameraWidget, cameraPosition);
        window.showWidget("CS", coordinateSystem);
        window.showWidget("cloud", cloud);
        window.spinOnce(1, true);
    }
}
