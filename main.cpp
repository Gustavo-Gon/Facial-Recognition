#include <iostream>
#include <map>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;
namespace fs = std::filesystem;

const string dataPath = "/Users/gustavogonzalez/CLionProjects/FaceRec/images/";
const string faceCascadePath = "/Users/gustavogonzalez/CLionProjects/FaceRec/haarcascade_frontalface_default.xml";

map<string, int> nameToLabel = {
        {"Lexi", 0},
        {"Gus", 1},
        {"Kevin", 2}
};

vector<Mat> loadImagesAndLabels(const string& personName, vector<int>& labels) {
    vector<Mat> images;
    int label = nameToLabel[personName];

    for (const auto& entry : fs::directory_iterator(dataPath + personName)) {
        Mat image = imread(entry.path(), IMREAD_GRAYSCALE);
        if (!image.empty()) {
            images.push_back(image);
            labels.push_back(label);
        } else {
            cerr << "Error: Unable to load image from " << entry.path() << endl;
        }
    }
    return images;
}

void initFaceMap(map<int, string>& faceMap) {
    for (const auto& [name, label] : nameToLabel) {
        faceMap[label] = name;
    }
}

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Failed to open the webcam." << endl;
        return 1;
    }

    CascadeClassifier faceCascade;
    if (!faceCascade.load(faceCascadePath)) {
        cerr << "Error: Unable to load the face detection model." << endl;
        return 1;
    }

    map<int, string> faceMap;
    initFaceMap(faceMap);

    vector<Mat> faces;
    vector<int> labels;

    for (const auto& [name, _] : nameToLabel) {
        vector<Mat> personFaces = loadImagesAndLabels(name, labels);
        faces.insert(faces.end(), personFaces.begin(), personFaces.end());
    }

    if (faces.size() != labels.size() || faces.empty()) {
        cerr << "Error: Number of faces and labels mismatch or no data available." << endl;
        return 1;
    }

    double confidenceThreshold = 100.0;
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->train(faces, labels);

    Mat frame;
    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            cerr << "Error: Empty frame captured." << endl;
            continue;
        }

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        vector<Rect> detectedFaces;
        faceCascade.detectMultiScale(grayFrame, detectedFaces, 1.1, 5, 0, Size(50, 50));

        for (const Rect& face : detectedFaces) {
            Mat faceROI = grayFrame(face);
            if (faceROI.empty()) continue;

            int label = -1;
            double confidence = 0.0;
            recognizer->predict(faceROI, label, confidence);

            string personName = "Unknown";
            if (confidence <= confidenceThreshold && faceMap.find(label) != faceMap.end()) {
                personName = faceMap[label];
            }

            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, personName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
        }

        imshow("Facial Recognition", frame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
