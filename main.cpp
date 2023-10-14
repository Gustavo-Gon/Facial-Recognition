#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;
using namespace cv::face;

int main() {
    // Open the default webcam (you can specify a different camera index if needed)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Failed to open the webcam." << endl;
        return 1;
    }

    // Load the pre-trained face detection model
    CascadeClassifier faceCascade;
    if (!faceCascade.load("/Users/gustavogonzalez/CLionProjects/FaceRec/haarcascade_frontalface_default.xml")) {
        cerr << "Error: Unable to load the face detection model." << endl;
        return 1;
    }

    // Data structure to store face descriptors and corresponding person names
    map<int, string> faceMap;

    // Add face images and labels here
    vector<Mat> faces;
    vector<int> labels;

    // Load and label face images for recognition
    Mat lexi1 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Lexi.jpeg", IMREAD_GRAYSCALE);
    Mat lexi2 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Lexi2.JPG", IMREAD_GRAYSCALE);
    Mat lexi3 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Lexi3.JPG", IMREAD_GRAYSCALE);
    Mat lexi4 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Lexi4.JPG", IMREAD_GRAYSCALE);
    Mat lexi5 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Lexi5.JPG", IMREAD_GRAYSCALE);
    Mat gus = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Gus.JPG", IMREAD_GRAYSCALE);
    Mat gus2 =imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Gus2.JPG", IMREAD_GRAYSCALE);
    Mat gus3 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Gus3.JPG", IMREAD_GRAYSCALE);
    Mat gus4 =imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Gus4.JPG", IMREAD_GRAYSCALE);
    Mat gus5 = imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Gus5.JPG", IMREAD_GRAYSCALE);
    Mat gus6 =imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Gus6.JPG", IMREAD_GRAYSCALE);
    Mat kev =imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Kevin.jpeg", IMREAD_GRAYSCALE);
    Mat kev2 =imread("/Users/gustavogonzalez/CLionProjects/FaceRec/images/Kevin2.jpeg", IMREAD_GRAYSCALE);

    // Add the face images and corresponding labels to the vectors
    faces.push_back(lexi1);
    faces.push_back(lexi2);
    faces.push_back(lexi3);
    faces.push_back(lexi4);
    faces.push_back(lexi5);
    faces.push_back(gus);
    faces.push_back(gus2);
    faces.push_back(gus3);
    faces.push_back(gus4);
    faces.push_back(gus5);
    faces.push_back(gus6);
    faces.push_back(kev);
    faces.push_back(kev2);

    labels.push_back(0); // Lexi's label is 0
    labels.push_back(0); // Lexi's label is 0
    labels.push_back(0); // Lexi's label is 0
    labels.push_back(0); // Lexi's label is 0
    labels.push_back(0);
    labels.push_back(1); // Gus's label is 1
    labels.push_back(1);
    labels.push_back(1); // Gus's label is 1
    labels.push_back(1);
    labels.push_back(1); // Gus's label is 1
    labels.push_back(1);
    labels.push_back(2); // Kevin's label is 1
    labels.push_back(2);

    // Add corresponding names to the faceMap
    faceMap[0] = "Lexi";
    faceMap[1] = "Gus";
    faceMap[2] = "Kevin";

    // Set a confidence threshold (adjust as needed)
    double confidenceThreshold = 80.0;

    // Train the LBPHFaceRecognizer model with the training data
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->train(faces, labels);

    // Main loop for real-time face recognition
    Mat frame;
    while (cap.read(frame)) {
        // Convert the frame to grayscale (face detection works on grayscale images)
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Perform face detection on the frame
        vector<Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5, 0, Size(50, 50));

        // Process each detected face
        for (const Rect& face : faces) {
            // Crop the face region from the frame
            Mat faceROI = grayFrame(face);

            // Perform face recognition to get the face descriptor
            int label = -1;
            double confidence = 0.0;
            recognizer->predict(faceROI, label, confidence);

            // Retrieve the associated person name from the data structure
            string personName;
            if (confidence < confidenceThreshold) {
                personName = "Unknown";
            } else {
                if (faceMap.find(label) != faceMap.end()) {
                    personName = faceMap[label];
                } else {
                    personName = "Unknown";
                }
            }

            // Draw a rectangle around the detected face and display the person's name
            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, personName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
        }

        // Display the result
        imshow("Facial Recognition", frame);

        // Wait for a key press and exit the loop if the user presses the 'q' key
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the VideoCapture and close the window
    cap.release();
    destroyAllWindows();

    return 0;
}