#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


int main() {
    cv::Mat sourceL = cv::imread("/home/tibor/Desktop/opencv-projects/3DCoordsEstimation/epipolarImages/sourceL.jpg");
    cv::Mat sourceR = cv::imread("/home/tibor/Desktop/opencv-projects/3DCoordsEstimation/epipolarImages/sourceR.jpg");

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
            points2.push_back(keypoints2[pair[0].trainIdx].pt);
            points1.push_back(keypoints1[pair[1].queryIdx].pt);
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
    cv::computeCorrespondEpilines(points1, 1, fundamentalMatrix, epilinesSourceR);
    cv::computeCorrespondEpilines(points2, 2, fundamentalMatrix, epilinesSourceL);

    auto sourceSize = points1.size();

    // Draw epilines on the second image (sourceR)
    for (size_t i = 0; i < sourceSize; i++) {
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        // sourceR
        cv::line(sourceR, cv::Point(0, -epilinesSourceR[i][2] / epilinesSourceR[i][1]),
                 cv::Point(sourceR.cols, -(epilinesSourceR[i][2] + epilinesSourceR[i][0] * sourceR.cols)
                                         / epilinesSourceR[i][1]), color);
        cv::circle(sourceR, points2[i], 5, color, -1);
        // sourceL
        cv::line(sourceL, cv::Point(0, -epilinesSourceL[i][2] / epilinesSourceL[i][1]),
                 cv::Point(sourceL.cols, -(epilinesSourceL[i][2] + epilinesSourceL[i][0] * sourceL.cols)
                                         / epilinesSourceL[i][1]), color);
        cv::circle(sourceL, points1[i], 5, color, -1);
    }

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
            1097.1921, 0, 719.5,
            0, 1097.1921, 539.5,
            0, 0, 1
    );

    cv::Mat distortionCoefficients = (cv::Mat_<double>(1, 5) <<
           -0.116098, 0.0740844, 0, 0, 0
    );

    std::vector<cv::Point2f> undistortedPoints1;
    std::vector<cv::Point2f> undistortedPoints2;

    cv::undistortPoints(points1, undistortedPoints1, cameraMatrix, distortionCoefficients);
    cv::undistortPoints(points2, undistortedPoints2, cameraMatrix, distortionCoefficients);

    cv::Matx34d projectionMatrix1 = cv::Matx34d::eye();
    cv::Matx34d projectionMatrix2;
    cv::hconcat(cameraMatrix, cv::Mat::zeros(3, 1, CV_64F), projectionMatrix1);
    cv::hconcat(rotationVectors[1], translationalVectors[1], projectionMatrix2);

    cv::Mat points4D;
    cv::triangulatePoints(projectionMatrix1, projectionMatrix2, undistortedPoints1, undistortedPoints2, points4D);

    cv::imshow("L", sourceL);
    cv::imshow("R", sourceR);
    cv::waitKey();

    return 0;
}
