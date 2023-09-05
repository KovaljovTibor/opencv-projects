#include <opencv2/opencv.hpp>

// 1. Converting image to HSV space - easier to detect colour
// 2. Creating Trackbars - with this, we can get hmin, hmax, smin, smax, vmin, vmax
// 3. Collecting our color by inRange
// 4. Displaying results

cv::Mat imgHSV, mask;
int hmin = 0, smin = 0, vmin = 0;
int hmax = 179, smax = 255, vmax = 255;

cv::VideoCapture capture(0);
cv::Mat img;

int main() {
    cv::namedWindow("Trackbars", (640, 200));
    cv::createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    cv::createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    cv::createTrackbar("Sat Min", "Trackbars", &smin, 255);
    cv::createTrackbar("Sat Max", "Trackbars", &smax, 255);
    cv::createTrackbar("Val Min", "Trackbars", &vmin, 255);
    cv::createTrackbar("Val Max", "Trackbars", &vmax, 255);

    while (true) {
        capture.read(img);
        cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);
        cv::Scalar lowerLimit(hmin, smin, vmin);
        cv::Scalar upperLimit(hmax, smax, vmax);
        cv::inRange(imgHSV, lowerLimit, upperLimit, mask);
        cv::imshow("Image", img);
        cv::imshow("Image HSV", imgHSV);
        cv::imshow("Image Mask", mask);
        if (cv::waitKey(30) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}