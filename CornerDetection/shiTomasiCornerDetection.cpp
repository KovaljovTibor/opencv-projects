#include <iostream>
#include <opencv2/opencv.hpp>


void shiTomasiCornerDetection() {
    cv::Mat source, gray;
    cv::Mat output, outputNorm, outputNormScaled;

    source = cv::imread("/home/tibor/Desktop/opencv-projects/CornerDetection/house.jpg");
    cv::resize(source, source, cv::Size(640, 480));
    if (source.empty()) std::cout << "Could not load the image." << '\n';

    cv::imshow("Shi-Tomasi House sample", source);

    cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, 100, 0.01, 10, cv::Mat(), 3, false, 0.04);

    for (const auto& corner: corners) {
        cv::circle(source, corner, 4, cv::Scalar(0, 255, 0), 2, 8, 0);
    }

    cv::imshow("Shi-Tomasi House corners detection", source);

    cv::waitKey();
}


int main() {
    shiTomasiCornerDetection();
    return 0;
}