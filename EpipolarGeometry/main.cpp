#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

int main() {
    cv::Mat sourceL = cv::imread("/home/tibor/Desktop/opencv-projects/EpipolarGeometry/sourceL.jpg");
    cv::Mat sourceR = cv::imread("/home/tibor/Desktop/opencv-projects/EpipolarGeometry/sourceR.jpg");

    if (sourceL.empty() || sourceR.empty()) {
        std::cerr << "Error: Could not load the image." << '\n';
        return EXIT_FAILURE;
    }

    cv::Mat sourceLGray;
    cv::Mat sourceRGray;
    cv::cvtColor(sourceL, sourceLGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(sourceR, sourceRGray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat destination1;
    cv::Mat destination2;

    sift->detectAndCompute(sourceL, cv::Mat(), keypoints1, destination1);
    sift->detectAndCompute(sourceR, cv::Mat(), keypoints2, destination2);

    std::vector<std::vector<cv::DMatch>> matches;
    cv::Ptr<cv::flann::KDTreeIndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);

    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
    matcher->knnMatch(destination1, destination2, matches, 2);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (const auto& pair: matches) {
        if (pair[0].distance < pair[1].distance * 0.8) {
            points1.push_back(keypoints1[pair[0].queryIdx].pt);
            points2.push_back(keypoints2[pair[0].trainIdx].pt);
        }
    }

    cv::Mat mask;
    cv::Mat fundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_LMEDS, 3, 0.99, mask);

    std::vector<cv::Point2f> filteredPoints1;
    std::vector<cv::Point2f> filteredPoints2;

    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<uchar>(i) == 1) {
            filteredPoints1.push_back(points1[i]);
            filteredPoints2.push_back(points2[i]);
        }
    }

    points1 = filteredPoints1;
    points2 = filteredPoints2;

    // Compute epilines
    std::vector<cv::Vec3f> epilinesSourceR, epilinesSourceL;
    cv::computeCorrespondEpilines(points1, 2, fundamentalMatrix, epilinesSourceR);
    cv::computeCorrespondEpilines(points2, 1, fundamentalMatrix, epilinesSourceL);

    for (size_t i = 0; i < points1.size(); i++) {
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

        cv::line(sourceL, cv::Point(sourceR.cols, -epilinesSourceR[i][2] / epilinesSourceR[i][1]),
                 cv::Point(points1[i].x, points1[i].y), color, 1);
        cv::circle(sourceL, points1[i], 5, color, -1);

        cv::line(sourceR, cv::Point(0, -epilinesSourceL[i][2] / epilinesSourceL[i][1]),
                 cv::Point(points2[i].x, points2[i].y), color, 1);
        cv::circle(sourceR, points2[i], 5, color, -1);
    }

    cv::imshow("L", sourceL);
    cv::imshow("R", sourceR);
    cv::waitKey();

    return 0;
}