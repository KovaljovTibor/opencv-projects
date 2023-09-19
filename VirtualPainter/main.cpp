#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat image;
std::vector<std::vector<int>> newPoints;

std::vector<std::vector<int>> myColors = {
        {0, 104, 39, 76, 255, 255},     // green
        {0, 143, 51, 85, 255, 255}      // orange
};

std::vector<cv::Scalar> myColorValues = {
        {255, 165, 0},
        {255, 255, 0}
};


cv::Point getContours(const cv::Mat &imgDil) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgDil, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> contourPoly(contours);
    std::vector<cv::Rect> boundRect(contours.size());

    cv::Point myPoint(0, 0);

    for (int i = 0; i < contours.size(); i++) {
        auto area = cv::contourArea(contours[i]);
        std::cout << "Area: " << area << '\n';
        if (area > 1000) {
            auto peri = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], contourPoly[i], 0.02 * peri, true);
            std::cout << "Corner points: " << contourPoly[i].size() << '\n';
            boundRect[i] = cv::boundingRect(contourPoly[i]);

            myPoint.x = boundRect[i].x + boundRect[i].width / 2;
            myPoint.y = boundRect[i].y;

            cv::drawContours(image, contourPoly, i, cv::Scalar(255, 0, 255), 2);
            cv::rectangle(image, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 0), 5);
        }
    }

    return myPoint;
}


std::vector<std::vector<int>> findColor(const cv::Mat& img) {
    cv::Mat imgHSV;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    for (int i = 0; i < myColors.size(); i++) {
        cv::Scalar lowerLimit(myColors[i][0], myColors[i][1], myColors[i][2]);
        cv::Scalar upperLimit(myColors[i][3], myColors[i][4], myColors[i][5]);
        cv::Mat mask;
        cv::inRange(imgHSV, lowerLimit, upperLimit, mask);
        cv::imshow(std::to_string(i), mask);
        cv::Point myPoint = getContours(mask);
        if (myPoint.x != 0 && myPoint.y != 0) {
            newPoints.push_back({myPoint.x, myPoint.y, i});
        }
    }

    return newPoints;
}


void drawOnCanvas(const std::vector<std::vector<int>> &points, std::vector<cv::Scalar> colorValues) {
    for (int i = 0; i < points.size(); i++) {
        cv::circle(image, cv::Point(newPoints[i][0], newPoints[i][1]), 10, colorValues[points[i][2]], cv::FILLED);
    }
}


int main() {
    cv::VideoCapture capture(0);

    while (true) {
        capture.read(image);
        newPoints = findColor(image);
        drawOnCanvas(newPoints, myColorValues);
        cv::imshow("Image", image);
        if (cv::waitKey(30) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}