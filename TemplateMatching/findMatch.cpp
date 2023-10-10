#include <opencv2/opencv.hpp>


int main() {
    cv::Mat source = cv::imread("/home/tibor/Desktop/opencv-projects/TemplateMatching/f16.jpg");
    cv::Mat sourceGray;
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);
    cv::Mat templateImage = cv::imread("/home/tibor/Desktop/opencv-projects/TemplateMatching/f16_template.jpg",
                                       cv::IMREAD_GRAYSCALE);

    int height = templateImage.rows;
    int width = templateImage.cols;

    cv::Mat result;
    cv::matchTemplate(sourceGray, templateImage, result, cv::TM_SQDIFF);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Point topLeft = minLoc;
    cv::Point bottomRight(topLeft.x + width, topLeft.y + height);

    cv::rectangle(source, topLeft, bottomRight, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Matched image", source);
    cv::waitKey();

    return 0;
}