#pragma warning(push, 0)
#include<iostream>
#include<string>

#include<opencv2/calib3d.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/viz.hpp>
#pragma warning(pop)

#include"include/calibration/calibration.hpp"
#include"include/calibration/calibration_settings.hpp"
#include"include/loader.hpp"
#include"tracking/tracker.hpp"
#include"converter.hpp"

int WIDTH = 1080, HEIGHT = 720;

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
    slam::Calibration calibration(settings, false, WIDTH);
    slam::save(calibration, outputFile, "Calibration");
    return calibration;
}

std::tuple<cv::viz::WCameraPosition, cv::Affine3d>
cameraFromKeyFrame(std::shared_ptr<slam::KeyFrame> keyframe) {
    cv::Matx44f projection = keyframe->getPose();
    cv::viz::Camera camera(projection, cv::Size(WIDTH, HEIGHT));
    cv::Affine3d cameraPosition = cv::viz::makeCameraPose(
        keyframe->getCameraCenter(),
        cv::Vec3f(0.0f, 0.0f, 1.0f),
        cv::Vec3f(0.0f, 1.0f, 1.0f)
    );
    return {
        cv::viz::WCameraPosition(camera.getFov()), cameraPosition
    };
}

void drawMap(std::shared_ptr<slam::Map> map) {
    std::vector<std::tuple<cv::viz::WCameraPosition, cv::Affine3d>> cameras;
    for (const auto& keyframe : map->getKeyframes())
        cameras.push_back(cameraFromKeyFrame(keyframe));

    /* Visualization */
    std::vector<cv::Point3f> adjustedPoints;
    for (const auto& p : map->getMappoints())
        adjustedPoints.push_back(p->getWorldPos());

    cv::viz::Viz3d window("slam");
    cv::viz::WCloud cloud(adjustedPoints, cv::viz::Color::red());
    /* cv::viz::WCoordinateSystem coordinateSystem; */
    int cameraId = 0;
    while (!window.wasStopped()) {
        for (const auto& [cameraWidget, cameraPosition] : cameras) {
            window.showWidget(
                std::to_string(cameraId++) + "CW",
                cameraWidget, cameraPosition
            );
        }
        /* window.showWidget("CS", coordinateSystem); */
        window.showWidget("cloud", cloud);
        window.spinOnce(1, true);
    }
}

int main() {
    auto calibration = getCalibration(false);
    auto detector = std::shared_ptr<slam::Detector>(new slam::Detector(
        cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::FAST_SCORE)
    ));
    slam::Tracker tracker(calibration, detector);

    cv::VideoCapture capture("C:\\Users\\tonys\\Downloads\\cupp.mp4");
    if (!capture.isOpened()) {
        std::cerr << "Cannot open file" << std::endl;
        return -1;
    }
    int n = 0;
    int i = 0;
    while (true) {
        cv::Mat frame;
        capture >> frame;
        if (frame.empty()) break;

        if (i++ % 35 != 0) continue;
        i = 1;

        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        float scale = (
            static_cast<float>(WIDTH)
            / static_cast<float>(frame.size().width)
        );
        cv::resize(frame, frame, cv::Size(0, 0), scale, scale);

        imshow("henlo", frame);
        if (cv::waitKey(1) == 'q') break;

        auto currentImage = std::make_shared<cv::Mat>(frame.clone());
        tracker.track(currentImage);

        if (n++ == 8) break;
        /* if (tracker.state == slam::Tracker::INITIALIZED) break; */
    }

    capture.release();
    cv::destroyAllWindows();

    drawMap(tracker.map);
}
