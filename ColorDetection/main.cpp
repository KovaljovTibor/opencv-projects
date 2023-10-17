#include <opencv2/opencv.hpp>


cv::Mat sourceHSV;
cv::Mat mask;
int hueMin = 0;
int satMin = 110;
int valMin = 153;
int hueMax = 19;
int satMax = 240;
int valMax = 255;


int main() {
    cv::Mat source = cv::imread("/home/tibor/Desktop/opencv-projects/ColorDetection/lambo.png");
    cv::cvtColor(source, sourceHSV, cv::COLOR_BGR2HSV);

    cv::namedWindow("Trackbars");
    int windowWidth = 640;
    int windowHeight = 200;
    cv::resizeWindow("Trackbars", cv::Size(windowWidth, windowHeight));

    cv::createTrackbar("Hue Minimum", "Trackbars", &hueMin, 179);
    cv::createTrackbar("Hue Maximum", "Trackbars", &hueMax, 179);
    cv::createTrackbar("Saturation Minimum", "Trackbars", &satMin, 255);
    cv::createTrackbar("Saturation Maximum", "Trackbars", &satMax, 255);
    cv::createTrackbar("Value Minimum", "Trackbars", &valMin, 255);
    cv::createTrackbar("Value Maximum", "Trackbars", &valMax, 255);

    while (cv::waitKey(30) != 27) {
        cv::Scalar lowerLimit(hueMin, satMin, valMin);
        cv::Scalar upperLimit(hueMax, satMax, valMax);

        cv::inRange(sourceHSV, lowerLimit, upperLimit, mask);

        cv::imshow("Source", source);
        cv::imshow("Source HSV", sourceHSV);
        cv::imshow("Image Mask", mask);
    }

    return 0;
}