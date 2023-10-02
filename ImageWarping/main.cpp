#include <opencv2/opencv.hpp>


int width = 250;
int height = 350;
cv::Mat matrix;
cv::Mat imageWarp;


int main() {
    cv::Mat image = cv::imread("/home/tibor/Desktop/opencv-projects/ImageWarping/cards.jpg");

    cv::Point2f kingSourcePoints[4] = {
            {529, 142}, {771, 190}, {405, 395}, {674, 457}
    };
    cv::Point2f queenSourcePoints[4] = {
            {529, 142}, {771, 190}, {405, 395}, {674, 457}
    };
    cv::Point2f destinationPoints[4] = {
            {0.0f, 0.0f}, {(float)width, 0.0f}, {0.0f, (float)height}, {(float)width, (float)height}
    };

    matrix = cv::getPerspectiveTransform(kingSourcePoints, destinationPoints);
    cv::warpPerspective(image, imageWarp, matrix, cv::Point(width, height));

    for (const auto& kingSourcePoint : kingSourcePoints) {
        cv::circle(image, kingSourcePoint, 10, cv::Scalar(0, 69, 255), cv::FILLED);
    }

    cv::imshow("Image", image);
    cv::imshow("Image Warped", imageWarp);
    cv::waitKey();

    return 0;
}