#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


int main() {
    cv::Mat sourceL = cv::imread("/home/tibor/Desktop/opencv-projects/EpipolarGeometry/headA.jpg");
    cv::Mat sourceR = cv::imread("/home/tibor/Desktop/opencv-projects/EpipolarGeometry/headB.jpg");

    if (sourceL.empty() || sourceR.empty()) {
        std::cerr << "Error: Could not load the image." << '\n';
        return EXIT_FAILURE;
    }

    cv::Mat sourceLGray;
    cv::Mat sourceRGray;
    cv::cvtColor(sourceL, sourceLGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(sourceR, sourceRGray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    std::vector<cv::KeyPoint> keypoint1;
    std::vector<cv::KeyPoint> keypoint2;
    cv::Mat destination1;
    cv::Mat destination2;

    sift->detectAndCompute(sourceL, cv::Mat(), keypoint1, destination1);
    sift->detectAndCompute(sourceR, cv::Mat(), keypoint2, destination2);

    std::vector<std::vector<cv::DMatch>> matches;
    cv::Ptr<cv::flann::KDTreeIndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>();
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);

    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
    matcher->knnMatch(destination1, destination2, matches, 2);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (const auto& pair: matches) {
        if (pair[0].distance < pair[1].distance * 0.8) {
            points2.push_back(keypoint2[pair[0].trainIdx].pt);
            points1.push_back(keypoint1[pair[1].queryIdx].pt);
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
    std::vector<cv::Vec3f> epilines1, epilines2;
    cv::computeCorrespondEpilines(points1, 1, fundamentalMatrix, epilines1);
    cv::computeCorrespondEpilines(points2, 2, fundamentalMatrix, epilines2);

    std::vector<cv::Scalar> colors;
    for (size_t i = 0; i < points1.size(); i++) {
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        colors.push_back(color);
    }

    auto sourceSize = points1.size();

    // Draw epilines on the second image (sourceR)
    for (size_t i = 0; i < sourceSize; i++) {
        cv::Scalar color = colors[i];
        // sourceR
        cv::line(sourceR, cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
                 cv::Point(sourceR.cols, -(epilines1[i][2] + epilines1[i][0] * sourceR.cols) / epilines1[i][1]), color);
        cv::circle(sourceR, points2[i], 5, color, -1);
        // sourceL
        cv::line(sourceL, cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
                 cv::Point(sourceL.cols, -(epilines2[i][2] + epilines2[i][0] * sourceL.cols) / epilines2[i][1]), color);
        cv::circle(sourceL, points1[i], 5, color, -1);
    }

    cv::imshow("L", sourceL);
    cv::imshow("R", sourceR);
    cv::waitKey();

    return 0;
}
