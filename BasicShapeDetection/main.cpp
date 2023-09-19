#include <vector>
#include <opencv2/opencv.hpp>

// 0. loading the image
// 1. preprocessing the image - using Canny edge detector
// 2. finding contours in the processed image
// 3. displaying the result

cv::Mat gray;
cv::Mat blur;
cv::Mat canny;
cv::Mat dilation;

void getContours(const cv::Mat &img, const cv::Mat &drawOnImage) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        auto area = cv::contourArea(contours[i]);
        std::vector<std::vector<cv::Point>> contourPoly(contours.size());
        std::vector<cv::Rect> boundRect(contours.size());
        std::string objectType;
        if (area > 1000) {
            auto perimeter = cv::arcLength(contours[i],  true);
            cv::approxPolyDP(contours[i], contourPoly[i], 0.02 * perimeter, true);
            cv::drawContours(drawOnImage, contourPoly, i, cv::Scalar(255, 0, 255), 2);
            std::cout << contourPoly[i].size() << '\n';
            boundRect[i] = cv::boundingRect(contourPoly[i]);
            cv::rectangle(drawOnImage, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 0), 5);
            auto objectCorner = contourPoly[i].size();
            if (objectCorner == 3) { objectType = "Triangle"; }
            if (objectCorner == 4) { objectType = "Rectangle/Square"; }
            if (objectCorner > 4) { objectType = "Circle"; }
            cv::putText(drawOnImage, objectType, {boundRect[i].x, boundRect[i].y - 5},
                            cv::FONT_HERSHEY_PLAIN, 0.75, cv::Scalar(0, 69, 255), 1);
        }
    }
}


int main() {
    /* 0 */
    cv::Mat source = cv::imread("/home/tibor/Desktop/opencv-projects/resources/shapes.png");

    /* 1. */
    cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 3, 0);
    cv::Canny(blur, canny, 25, 75);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(canny, dilation, kernel);

    /* 2. */
    getContours(dilation, source);

    /* 3. */
    cv::imshow("Image", source);

    cv::waitKey(0);

    return 0;
}