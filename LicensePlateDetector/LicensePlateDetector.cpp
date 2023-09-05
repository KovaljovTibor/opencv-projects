#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture capture(0);
    cv::Mat img;

    std::string path = "/home/tibor/Desktop/tutorials/opencv/resources/haarcascade_russian_plate_number.xml";
    cv::CascadeClassifier plateDetection;
    plateDetection.load(path);
    std::vector<cv::Rect> plates;
    if (plateDetection.empty()) {
        std::cout << "XML not loaded." << '\n';
    }

    while (true) {
        capture.read(img);

        plateDetection.detectMultiScale(img, plates, 1.1, 10);
        for (int i = 0; i < plates.size(); i++) {
            cv::Mat imgCrop = img(plates[i]);
            cv::imwrite("/home/tibor/Desktop/tutorials/opencv/resources/Plates/" + std::to_string(i) + ".png", imgCrop);
            cv::rectangle(img, plates[i].tl(), plates[i].br(), cv::Scalar(255, 0, 255), 3);
        }

        cv::imshow("Image", img);
        if (cv::waitKey(30) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}