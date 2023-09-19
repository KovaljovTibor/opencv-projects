#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

cv::Mat imgOriginal;
cv::Mat imgGray;
cv::Mat imgBlur;
cv::Mat imgCanny;
cv::Mat imgThreshold;
cv::Mat imgDil;
cv::Mat imgWarp;

float width = 420;
float height = 596;

std::vector<cv::Point> initialPoints;
std::vector<cv::Point> documentPoints;

cv::Mat preprocessing(const cv::Mat &img) {
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(imgGray, imgBlur, cv::Size(3, 3), 3, 0);
    cv::Canny(imgBlur, imgCanny, 25, 75);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(imgCanny, imgDil, kernel);
    return imgDil;
}


std::vector<cv::Point> getContours(const cv::Mat &img) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> contourPoly(contours);
    std::vector<cv::Rect> boundRect(contours.size());

    std::vector<cv::Point> biggest;
    int maxArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        auto area = cv::contourArea(contours[i]);
        std::cout << "Area: " << area << '\n';
        if (area > 1000) {
            auto peri = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], contourPoly[i], 0.02 * peri, true);
            if (area > maxArea && contourPoly[i].size() == 4) {
//                cv::drawContours(imgOriginal, contourPoly, i, cv::Scalar(255, 0, 255), 5);
                biggest = {contourPoly[i][0], contourPoly[i][1], contourPoly[i][2], contourPoly[i][3]};
                maxArea = (int) area;
            }
        }
    }

    return biggest;
}


void drawPoints(std::vector<cv::Point> points, const cv::Scalar &color) {
    for (int i = 0; i < points.size(); i++) {
//        cv::circle(imgOriginal, points[i], 10, color, cv::FILLED);
//        cv::putText(imgOriginal, std::to_string(i), points[i], cv::FONT_HERSHEY_PLAIN, 4, color, 4);
    }
}


std::vector<cv::Point> reorder(const std::vector<cv::Point> &points) {
    std::vector<cv::Point> newPoints;
    std::vector<int> pointsSum;
    std::vector<int> pointsSub;

    for (int i = 0; i < 4; i++) {
        pointsSum.push_back(points[i].x + points[i].y);
        pointsSub.push_back(points[i].x - points[i].y);
    }

    newPoints.push_back(points[std::min_element(pointsSum.begin(), pointsSum.end()) - pointsSum.begin()]);
    newPoints.push_back(points[std::max_element(pointsSub.begin(), pointsSub.end()) - pointsSub.begin()]);
    newPoints.push_back(points[std::min_element(pointsSub.begin(), pointsSub.end()) - pointsSub.begin()]);
    newPoints.push_back(points[std::max_element(pointsSum.begin(), pointsSum.end()) - pointsSum.begin()]);

    return newPoints;
}


cv::Mat getWarp(const cv::Mat &img, const std::vector<cv::Point> &points, float w, float h) {
    cv::Point2f source[4] = {points[0], points[1], points[2], points[3]};
    cv::Point2f destination[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };
    cv::Mat matrix = cv::getPerspectiveTransform(source, destination);
    cv::warpPerspective(img, imgWarp, matrix, cv::Point((int) w, (int) h));
    return imgWarp;
}


int main() {
    std::string path = "/home/tibor/Desktop/opencv-projects/DocumentScanner/paper.jpg";
    imgOriginal = cv::imread(path);
    cv::resize(imgOriginal, imgOriginal, cv::Size(), 0.5, 0.5);

    // Preprocessing
    imgThreshold = preprocessing(imgOriginal);

    // Getting the biggest contours we can find
    initialPoints = getContours(imgThreshold);
    documentPoints = reorder(initialPoints);
    drawPoints(documentPoints, cv::Scalar(0, 255, 0));

    /* Warp */
    imgWarp = getWarp(imgOriginal, documentPoints, width, height);

    /* Displaying results */
    cv::imshow("Image", imgOriginal);
    cv::imshow("Image Dilation", imgThreshold);
    cv::imshow("Image Warp", imgWarp);

    cv::waitKey(0);

    return 0;
}