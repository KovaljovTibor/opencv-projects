#include "dftImplementation.cpp"
#include <opencv2/opencv.hpp>


// 1.
void takeDFT(cv::Mat& source, cv::Mat& destination) {
    cv::Mat originalComplex[2] = {source, cv::Mat::zeros(source.size(), CV_32F)};
    cv::Mat dftReady;
    cv::merge(originalComplex, 2, dftReady);

    cv::Mat dftOfOriginal;
    cv::dft(dftReady, dftOfOriginal, cv::DFT_COMPLEX_OUTPUT);
    destination = dftOfOriginal;
}

// 3.
void recenterDFT(cv::Mat& source) {
    int centerX = source.cols / 2;
    int centerY = source.rows / 2;

    cv::Mat q1(source, cv::Rect(0, 0, centerX, centerY));
    cv::Mat q2(source, cv::Rect(centerX, 0, centerX, centerY));
    cv::Mat q3(source, cv::Rect(0, centerY, centerX, centerY));
    cv::Mat q4(source, cv::Rect(centerX, centerY, centerX, centerY));

    cv::Mat swapMap;

    q1.copyTo(swapMap);
    q4.copyTo(q1);
    swapMap.copyTo(q4);

    q2.copyTo(swapMap);
    q3.copyTo(q2);
    swapMap.copyTo(q3);
}

// 2.
void showDFT(cv::Mat& source) {
    cv::Mat splitArray[2] = {cv::Mat::zeros(source.size(), CV_32F), cv::Mat::zeros(source.size(), CV_32F)};
    cv::split(source, splitArray);

    cv::Mat dftMagnitude;
    cv::magnitude(splitArray[0], splitArray[1], dftMagnitude);

    dftMagnitude += cv::Scalar::all(1);
    cv::log(dftMagnitude, dftMagnitude);
    cv::normalize(dftMagnitude, dftMagnitude, 0, 1, cv::NORM_MINMAX);

    recenterDFT(dftMagnitude);
    cv::imshow("DFT", dftMagnitude);
    cv::waitKey();
}

// Returning our original image from DFT
void invertDFT(cv::Mat& source, cv::Mat& destination) {
    cv::Mat inverse;
    cv::dft(source, inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    destination = inverse;
}


int main() {
    cv::Mat original = cv::imread("/home/tibor/Desktop/opencv-projects/DiscreteFourierTransform/lenna.png");
    if (original.empty()) {
        std::cout << "Error: Image could not load." << '\n';
    }

    cv::cvtColor(original, original, cv::COLOR_BGR2GRAY);

    cv::Mat originalFloat;
    original.convertTo(originalFloat, CV_32FC1, 1.0 / 255.0);

    cv::Mat dftOfOriginal;

    takeDFT(originalFloat, dftOfOriginal);
    showDFT(dftOfOriginal);

    cv::Mat invertedDFT;
    invertDFT(dftOfOriginal, invertedDFT);

    cv::imshow("InvertDFT result", invertedDFT);
    cv::waitKey();

    return 0;
}