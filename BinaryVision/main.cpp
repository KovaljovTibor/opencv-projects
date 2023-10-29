#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


int main() {
    cv::Mat source = cv::imread("/home/tibor/Desktop/opencv-projects/BinaryVision/coins.png");
    cv::Mat sourceGray;
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);  // Convert to grayscale

    if (source.empty()) {
        std::cout << "Error: Could not load the image!";
        return EXIT_FAILURE;
    }

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat binaryImg;
    cv::Mat countoursImg;

    cv::threshold(sourceGray, binaryImg, 50, 255, cv::THRESH_BINARY);
    cv::findContours(binaryImg, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    cv::imshow("Original image", source);

    for (int contour = 0; contour < contours.size(); contour++) {
        cv::Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
        cv::drawContours(source, contours, contour, colour, cv::FILLED, 8, hierarchy);
    }

    cv::imshow("Contour image", source);
    cv::imshow("1", sourceGray);
    cv::waitKey();

    return 0;
}