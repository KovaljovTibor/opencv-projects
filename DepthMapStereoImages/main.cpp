#include <opencv2/opencv.hpp>


int main() {
    cv::Mat sourceL = cv::imread("/home/tibor/Desktop/opencv-projects/DepthMapStereoImages/source1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat sourceR = cv::imread("/home/tibor/Desktop/opencv-projects/DepthMapStereoImages/source2.png", cv::IMREAD_GRAYSCALE);
    cv::resize(sourceL, sourceL, cv::Size(600, 500));
    cv::resize(sourceR, sourceR, cv::Size(600, 500));

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
    cv::Mat disparity;
    stereo->compute(sourceL, sourceR, disparity);

    cv::imshow("SL", sourceL);
    cv::imshow("SR", sourceR);
    cv::imshow("Disparity Map", disparity);

    cv::waitKey();

    return 0;
}