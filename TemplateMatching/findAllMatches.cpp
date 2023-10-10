#include <opencv2/opencv.hpp>


int main() {
    cv::Mat source = cv::imread("/home/tibor/Desktop/opencv-projects/TemplateMatching/f16.jpg");
    cv::Mat sourceGray;
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);

    cv::Mat templateImage = cv::imread("/home/tibor/Desktop/opencv-projects/TemplateMatching/f16_template.jpg");
    cv::cvtColor(templateImage, templateImage, cv::COLOR_BGR2GRAY);

    int height = templateImage.rows;
    int width = templateImage.cols;

    cv::Mat result;
    cv::matchTemplate(sourceGray, templateImage, result, cv::TM_CCOEFF_NORMED);

    double threshold = 0.45;
    cv::Mat resultCopy = result.clone();

    while (true) {
        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;

        cv::minMaxLoc(resultCopy, &minVal, &maxVal, &minLoc, &maxLoc);

        if (maxVal >= threshold) {
            cv::Point topLeft = maxLoc;
            cv::Point bottomRight(topLeft.x + width, topLeft.y + height);
            cv::rectangle(source, topLeft, bottomRight, cv::Scalar(0, 0, 255), 1);
            cv::rectangle(resultCopy, maxLoc, cv::Point(maxLoc.x + width, maxLoc.y + height), cv::Scalar(0), -1);
        } else {
            break;
        }
    }

    cv::imshow("All matches", source);
    cv::waitKey();

    return 0;
}