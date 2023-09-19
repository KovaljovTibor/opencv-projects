#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    // Loading Video Capture
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cout << "Error: Webcam could not open." << '\n';
        return 1;
    }

    // Setting up pretrained model
    std::string path = "/home/tibor/Desktop/opencv-projects/resources/haarcascade_frontalface_default.xml";
    cv::CascadeClassifier pretrainedModel;
    if (!pretrainedModel.load(path)) {
        std::cerr << "Error: Pretrained model could not open." << '\n';
        return 1;
    }

    // Setting up containers
    cv::Mat img;
    cv::Mat imgGray;

    while (true) {
        capture.read(img);
        if (img.empty()) {
            std::cout << "Error: Image could not open." << '\n';
            break;
        }

        // Detecting face operation
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> coordinateList;
        pretrainedModel.detectMultiScale(imgGray, coordinateList, 1.1, 3);

        // Drawing rectangle on the face
        for (const cv::Rect &rect: coordinateList) {
            cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
        }

        // Displaying result
        cv::imshow("Webcam", img);
        if (cv::waitKey(30) == 27) {
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}