#include <opencv2/opencv.hpp>

int main() {
    /* 1D Histograms */

    cv::Mat sourceGray = cv::imread("/home/tibor/Desktop/opencv-projects/CornerDetection/house.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(sourceGray, sourceGray, {640, 480});

    cv::MatND histogram;
    int histSize = 256;
    const int* channelNumbers = nullptr;
    float channelRange[] = {0.0, 256.0};
    const float* channelRanges = channelRange;
    int numberBins = histSize;

    cv::calcHist(&sourceGray, 1, nullptr, cv::Mat(), histogram, 1, &numberBins, &channelRanges);

    int histW = 512;
    int histH = 400;
    int binW = cvRound((double) histW / histSize);
    cv::Mat histImage(histH, histW, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(binW * (i - 1), histH - cvRound(histogram.at<float>(i - 1))),
                 cv::Point(binW * (i), histH - cvRound(histogram.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    cv::imshow("Gray", sourceGray);
    cv::imshow("Gray Histogram", histImage);


    /* Color histograms */

    cv::Mat sourceColor = cv::imread("/home/tibor/Desktop/opencv-projects/CornerDetection/house.jpg", cv::IMREAD_COLOR);
    cv::resize(sourceColor, sourceColor, {640, 480});

    std::vector<cv::Mat> bgrPlanes;
    cv::split(sourceColor, bgrPlanes);

    int histSizeColor = 256;
    float range[] = {0, 256};
    const float* histRangeColor = {range};
    bool uniform = true;
    bool accumulate = false;

    cv::Mat bHist, gHist, rHist;

    cv::calcHist(&bgrPlanes[0], 1, nullptr, cv::Mat(), bHist, 1, &histSizeColor, &histRangeColor, uniform, accumulate);
    cv::calcHist(&bgrPlanes[1], 1, nullptr, cv::Mat(), gHist, 1, &histSizeColor, &histRangeColor, uniform, accumulate);
    cv::calcHist(&bgrPlanes[2], 1, nullptr, cv::Mat(), rHist, 1, &histSizeColor, &histRangeColor, uniform, accumulate);

    int histWColor = 512, histHColor = 400;
    int binWColor = cvRound((double) histW / histSizeColor);

    cv::Mat histImageColor(histHColor, histWColor, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(bHist, bHist, 0, histImageColor.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(gHist, gHist, 0, histImageColor.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(rHist, rHist, 0, histImageColor.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 0; i < histSize; i++) {
        cv::line(histImageColor, cv::Point(binWColor * (i - 1), histHColor - cvRound(bHist.at<float>(i - 1))),
                 cv::Point(binWColor * (i), histHColor - cvRound(bHist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImageColor, cv::Point(binWColor * (i - 1), histHColor - cvRound(gHist.at<float>(i - 1))),
                 cv::Point(binWColor * (i), histHColor - cvRound(gHist.at<float>(i))),
                 cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(histImageColor, cv::Point(binWColor * (i - 1), histHColor - cvRound(rHist.at<float>(i - 1))),
                 cv::Point(binWColor * (i), histHColor - cvRound(rHist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    cv::imshow("Color", sourceColor);
    cv::imshow("Color Histogram", histImageColor);

    cv::waitKey();

    return 0;
}