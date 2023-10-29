#include <opencv2/opencv.hpp>


int main() {
    cv::Mat sourceL = cv::imread("/home/tibor/Desktop/opencv-projects/BasicStereoVision/castleA.png");
    cv::Mat sourceR = cv::imread("/home/tibor/Desktop/opencv-projects/BasicStereoVision/castleB.png");

    cv::imshow("L", sourceL);
    cv::imshow("R", sourceR);

    int minDisparity = 0;
    int numDisparities = 64;
    int blockSize = 8;
    int disp12MaxDiff = 1;
    int uniquenessRatio = 10;
    int speckleWindowSize = 10;
    int speckleRange = 8;

    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, disp12MaxDiff,
                                                            uniquenessRatio, speckleWindowSize, speckleRange);

    cv::Mat disparity;
    stereo->compute(sourceL, sourceR, disparity);
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::imshow("Disparity", disparity);
    cv::waitKey();

    return 0;
}