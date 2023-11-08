#include <opencv2/opencv.hpp>


int main() {
    cv::Mat sourceL = cv::imread("/home/tibor/Desktop/opencv-projects/DepthMapStereoImages/castleA.png");
    cv::Mat sourceR = cv::imread("/home/tibor/Desktop/opencv-projects/DepthMapStereoImages/castleB.png");

    cv::cvtColor(sourceL, sourceL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(sourceR, sourceR, cv::COLOR_BGR2GRAY);
    cv::resize(sourceL, sourceL, cv::Size(600, 500));
    cv::resize(sourceR, sourceR, cv::Size(600, 500));

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    int numDisparities = 16;
    int blockSize = 15;

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities, blockSize);
    cv::Mat disparity;
    stereo->compute(sourceL, sourceR, disparity);

    cv::imshow("SL", sourceL);
    cv::imshow("SR", sourceR);
    cv::imshow("Disparity Map", disparity);

    cv::waitKey();

    return 0;
}