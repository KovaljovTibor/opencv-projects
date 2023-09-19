#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::string path ="/home/tibor/Desktop/opencv-projects/resources/color-segmentation/rectangle.png";
    cv::Mat img = cv::imread(path);
    cv::Mat greenShow(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

    cv::Mat green(img.rows * img.cols, 3, CV_32FC1, cv::Scalar(0));
    int idx = 0;
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            cv::Vec3b pixelValue = img.at<cv::Vec3b>(x, y);
            float sumRgb = pixelValue[0] + pixelValue[1] + pixelValue[2];
            if (sumRgb > 0.0) {
                float g = pixelValue[1] / sumRgb;
                green.at<cv::Vec3f>(idx, 0) = {g, g, g};
                greenShow.at<uchar>(x, y) = g * 255;
            }
            idx++;
        }
    }

    cv::imshow("Green", greenShow);

    cv::Ptr<cv::ml::EM> em = cv::ml::EM::create();
    em->setClustersNumber(8);

    cv::TermCriteria termCrit = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1.0);
    em->setTermCriteria(termCrit);

    em->trainEM(green);

    cv::Mat means = em->getMeans();
    cv::Mat weights = em->getWeights();

    std::vector<cv::Mat> images;
    for (size_t i = 0; i < 8; i++) {
        images.push_back(cv::Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0)));
    }

    idx = 0;
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            const int clusterId = cvRound(em->predict2(green.row(idx++), cv::noArray())[1]);
            images[clusterId].at<cv::Vec3b>(x, y) = img.at<cv::Vec3b>(x, y);
        }
    }

    cv::Mat field(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::bitwise_or(images[0], images[1], field);
    cv::bitwise_or(field, images[3], field);
    cv::bitwise_or(field, images[5], field);
    cv::bitwise_or(field, images[6], field);

    cv::Mat background(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::bitwise_or(images[2], images[4], background);
    cv::bitwise_or(background, images[7], background);

    cv::imshow("Field", field);
    cv::imshow("Background", background);

    cv::waitKey(0);
    return 0;
}