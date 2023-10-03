#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <fstream>
#include <vector>


const float calibrationSquareDimension = 0.02650f;
const float arucoSquareDimension = 0.26800f;
const cv::Size chessboardDimensions = cv::Size(6, 9);


void CreateArucoMarkers() {
    cv::Mat outputMarker;
    cv::Ptr<cv::aruco::Dictionary> markerDictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    for (int i = 0; i < 50; i++) {
        cv::aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
        std::string imageName = "/home/tibor/Desktop/opencv-projects/CameraCalibration/4x4Marker_";
        std::ostringstream converter;
        converter << imageName << i << ".jpg";
        cv::imwrite(converter.str(), outputMarker);
    }
}


void CreateKnownBoardPosition(cv::Size& boardSize, float squareEdgeLength, std::vector<cv::Point3f>& points) {
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            points.emplace_back((float)j * squareEdgeLength, (float)i * squareEdgeLength, 0.0f);
        }
    }
}


void GetIntersections(std::vector<cv::Mat>& images, std::vector<std::vector<cv::Point2f>>& allFoundPoints,
                          bool showResults = false) {
    for (auto iter = images.begin(); iter != images.end(); iter++) {
        std::vector<cv::Point2f> pointBuffer;
        bool found = cv::findChessboardCorners(*iter, chessboardDimensions, pointBuffer,
                                  cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            allFoundPoints.push_back(pointBuffer);
        }
        if (showResults) {
            cv::drawChessboardCorners(*iter, chessboardDimensions, pointBuffer, found);
            cv::imshow("Looking for points/intersections.", *iter);
            cv::waitKey();
        }
    }
}


int main() {
    cv::Mat frame;                                              // Classic captured frame from webcam video
    cv::Mat drawToFrame;                                        // Here, we copy proccessed frames
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);          // Identity matrix
    cv::Mat distanceCoefficients;

    std::vector<cv::Mat> savedImages;                           // Here, we save good calibrations
    std::vector<std::vector<cv::Point2f>> markerPoints;         // Points, that are found
    std::vector<std::vector<cv::Point2f>> rejectedCandidates;   // Points, which failed to meet to mark

    cv::VideoCapture capture(0);                                // Starting webcam
    if (!capture.isOpened()) {
        std::cout << "Error: Webcam could not open!" << '\n';
        return 0;
    }

    int fps = 20;                                               // Initializing frames per second
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);             // Initializing window for out webcam

    while (true) {
        if (!capture.read(frame)) break;
        std::vector<cv::Vec2f> foundPoints;
        bool found;
        found = cv::findChessboardCorners(frame, chessboardDimensions, foundPoints,
                                          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        cv::drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);

        if (found) {
            cv::imshow("Webcam", drawToFrame);
        } else {
            cv::imshow("Webcam", frame);
        }
        char character = cv::waitKey(1000 / fps);
    }

    return 0;
}