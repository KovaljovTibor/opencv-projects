#include <opencv2/opencv.hpp>
#include <iostream>


void harrisCornerDetector() {
    /* preparing sources and outputs */
    cv::Mat sample1, sample2, sampleGray1, sampleGray2;
    cv::Mat output1, output2, outputNorm1, outputNorm2, outputNormScaled1, outputNormScaled2;

    /* reading and preparing pictures */
    sample1 = cv::imread("/home/tibor/Desktop/opencv-projects/CornerDetection/cube.jpeg");
    cv::resize(sample1, sample1, cv::Size(500, 500));
    sample2 = cv::imread("/home/tibor/Desktop/opencv-projects/CornerDetection/house.jpg");
    cv::resize(sample2, sample2, cv::Size(640, 480));

    /* checking if the image is loaded successfuly */
    if (sample1.empty()) std::cout << "Could not load the image." << '\n';
    if (sample2.empty()) std::cout << "Could not load the image." << '\n';

    /* displaying starting pictures */
    cv::imshow("Harris Cube sample", sample1);
    cv::imshow("Harris House sample", sample2);

    /* converting calibrationImages(sources) into grayscale - easier to work with */
    cv::cvtColor(sample1, sampleGray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(sample2, sampleGray2, cv::COLOR_BGR2GRAY);

    /* preparing containers where we save the output */
    output1 = cv::Mat::zeros(sampleGray1.size(), CV_32FC1);
    output2 = cv::Mat::zeros(sampleGray2.size(), CV_32FC1);

    /* applying corner detector onto our input calibrationImages(sources) */
    cv::cornerHarris(sampleGray1, output1, 3, 3, 0.04);
    cv::cornerHarris(sampleGray2, output2, 3, 3, 0.04);

    /* normalizes the values, so it's easier to work with them */
    cv::normalize(output1, outputNorm1, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::normalize(output2, outputNorm2, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    /* scales, calculates absolute values, and converts the result to 8-bit */
    cv::convertScaleAbs(outputNorm1, outputNormScaled1);
    cv::convertScaleAbs(outputNorm2, outputNormScaled2);

    /* drawing circles on the corners */
    for (int r = 0; r < outputNorm1.rows; r++) {
        for (int c = 0; c < outputNorm2.cols; c++) {
            if ((int) outputNorm1.at<float>(r, c) > 100) {
                cv::circle(sample1, cv::Point(c, r), 2, cv::Scalar(0, 0, 255), 1, 8, 0);
            }
        }
    }

    for (int r = 0; r < outputNorm2.rows; r++) {
        for (int c = 0; c < outputNorm2.cols; c++) {
            if ((int) outputNorm2.at<float>(r, c) > 100) {
                cv::circle(sample2, cv::Point(c, r), 2, cv::Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }

    /* displaying results */
    cv::imshow("Cube sample corners detected", sample1);
    cv::imshow("House sample corners detected", sample2);

    cv::waitKey();
}


int main() {
    harrisCornerDetector();
    return 0;
}