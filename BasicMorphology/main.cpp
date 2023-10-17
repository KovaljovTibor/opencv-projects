#include <opencv2/opencv.hpp>


int main() {
    cv::Mat source = cv::imread("/home/tibor/Desktop/opencv-projects/BasicMorphology/source.png");
    cv::cvtColor(source, source, cv::COLOR_BGR2GRAY);
    if (source.empty()) std::cerr << "Error: Could not load the image!" << '\n';

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    cv::Mat erosionResult;
    cv::erode(source, erosionResult, kernel);

    cv::Mat dilationResult;
    cv::dilate(source, dilationResult, kernel);

    // opening - erosion followed by dilataion
    cv::Mat openingResult;
    cv::morphologyEx(source, openingResult, cv::MORPH_OPEN, kernel);

    cv::Mat closingResult;
    cv::morphologyEx(source, closingResult, cv::MORPH_CLOSE, kernel);

    // calculate gradient
    cv::Mat gradientResult;
    cv::morphologyEx(source, gradientResult, cv::MORPH_GRADIENT, kernel);
    
    // tophat (opening input)
    cv::Mat tophatResult;
    cv::morphologyEx(source, tophatResult, cv::MORPH_TOPHAT, kernel);
    
    // displaying result
    cv::imshow("Original", source);
    cv::imshow("Erosion", erosionResult);
    cv::imshow("Dilation", dilationResult);
    cv::imshow("Opening", openingResult);
    cv::imshow("Closing", closingResult);
    cv::imshow("Gradient", gradientResult);
    cv::imshow("Tophat", tophatResult);

    cv::waitKey();

    return 0;
}