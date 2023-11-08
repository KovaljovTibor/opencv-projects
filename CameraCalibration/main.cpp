#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main() {
    std::vector<cv::String> filenames;
    cv::glob("/home/tibor/Desktop/opencv-projects/CameraCalibration/calibrationImages/*.jpg", filenames, false);

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

    std::vector<double> perViewError;

    int cameraCalibrationFlags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 + cv::CALIB_ZERO_TANGENT_DIST
                                 + cv::CALIB_FIX_PRINCIPAL_POINT;

    cv::Size imageSize(1440, 1080);

    std::cout << "Calibrating..." << '\n';
    double error = cv::calibrateCamera(chessboardCorners3D, foundChessboardCorners, imageSize, cameraMatrix,
                                       distortionCoefficients, rotationVectors, translationalVectors,
                                       cameraCalibrationFlags);

    std::cout << "Reprojection error: " << error << "\nCamera Matrix: \t\n" << cameraMatrix
              << "Distortion Coefficients: \t\n" << distortionCoefficients
              << "\n\n";

    cv::Mat mapX;
    cv::Mat mapY;

    cv::initUndistortRectifyMap(cameraMatrix, distortionCoefficients, cv::Matx33f::eye(), cameraMatrix, imageSize,
                                CV_32FC1, mapX, mapY);

    for (const auto& filename: filenames) {
        std::cout << std::string(filename) << '\n';

        cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
        cv::Mat imgUndistorted;

        cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);
        cv::imshow("Undistorted image", imgUndistorted);
        cv::waitKey();
    }

    return 0;
}