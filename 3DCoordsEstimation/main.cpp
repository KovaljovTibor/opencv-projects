#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main() {
    /* CAMERA CALIBRATION */

    std::vector<cv::String> filenames;
    cv::glob("/home/tibor/Desktop/opencv-projects/3DCoordsEstimation/calibrationImages/*.jpg", filenames, false);

    cv::Size patternSize(10 - 1, 7 - 1);
    std::vector<std::vector<cv::Point2f>> foundChessboardCorners(filenames.size());
    std::vector<std::vector<cv::Point3f>> chessboardCorners3D;
    int checkerboard[2] = {10, 7};
    double fieldSize = 26.5;

    std::vector<cv::Point3f> worldPoints;
    for (int i = 1; i < checkerboard[1]; i++) {
        for (int j = 1; j < checkerboard[0]; j++) {
            worldPoints.emplace_back(j * fieldSize, i * fieldSize, 0);
        }
    }

    std::vector<cv::Point2f> imagePoints;

    std::size_t i = 0;
    for (const auto& filename: filenames) {
        std::cout << std::string(filename) << '\n';
        cv::Mat img = cv::imread(filenames[i]);

        cv::Mat imgGray;
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        int flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
        bool patternFound = cv::findChessboardCorners(imgGray, patternSize, foundChessboardCorners[i], flags);

        if (patternFound) {
            cv::cornerSubPix(imgGray, foundChessboardCorners[i], cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            chessboardCorners3D.push_back(worldPoints);
        }

        cv::drawChessboardCorners(img, patternSize, foundChessboardCorners[i], patternFound);
        cv::imshow("Chessboard Detection", img);
        cv::waitKey();

        i++;
    }

    cv::Matx33f cameraMatrix(cv::Matx33f::eye());
    cv::Vec<float, 5> distortionCoefficients(0, 0, 0, 0, 0);

    std::vector<cv::Mat> rotationVectors;
    std::vector<cv::Mat> translationalVectors;

    std::vector<double> stdIntrinsics;
    std::vector<double> stdExtrinsics;
    std::vector<double> perViewError;

    int cameraCalibrationFlags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST
                                 + cv::CALIB_FIX_PRINCIPAL_POINT;

    cv::Size frameSize(1440, 1080);

    std::cout << "Calibrating..." << '\n';
    double error = cv::calibrateCamera(chessboardCorners3D, foundChessboardCorners, frameSize, cameraMatrix,
                                       distortionCoefficients, rotationVectors, translationalVectors,
                                       cameraCalibrationFlags);

    std::cout << "Reprojection error: " << error << "\nCamera Matrix: \t\n" << cameraMatrix
              << "Distortion Coefficients: \t\n" << distortionCoefficients
              << "\n\n";

    cv::Mat mapX;
    cv::Mat mapY;

    cv::initUndistortRectifyMap(cameraMatrix, distortionCoefficients, cv::Matx33f::eye(), cameraMatrix, frameSize,
                                CV_32FC1, mapX, mapY);

    for (const auto& filename: filenames) {
        std::cout << std::string(filename) << '\n';

        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat imgUndistorted;

        cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);
        cv::imshow("Undistorted image", imgUndistorted);
        cv::waitKey();
    }


    /* EPIPOLAR LINES */

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


    /* 3D RECONSTRUCTION */

    cv::Mat projectionMat1;
    cv::Mat projectionMat2;

    std::vector<cv::Point3f> reconstructedPoints;

    /*
     * TODO - Compute projectionMat1, projectionMat2
     * TODO - Triangulate points found on the images
     * TODO - Compute X Y Z coordinates from the found points and print it out
    */

    return 0;
}