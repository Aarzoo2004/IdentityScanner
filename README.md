# IdentityScanner

A real-time **Age and Gender Detection** system using **CNN, SVM, and OpenCV**.

## 🚀 Features

* Detects faces in real-time from webcam feed
* Predicts **Age Range** and **Gender** of detected faces
* Uses **CNN for feature extraction** + **SVM for classification**
* Built with **OpenCV Deep Learning (dnn)** module
* Saves annotated video output

## 🛠️ Tech Stack

* **Python**
* **OpenCV** (face detection, video capture)
* **Convolutional Neural Network (CNN)** for feature extraction
* **Support Vector Machine (SVM)** for classification
* **Pre-trained OpenCV Models** (Caffe & TensorFlow formats)

## 📂 Project Structure

```
IdentityScanner/
│── main.py                     # Main script (run this)
│── age_deploy.prototxt         # Age detection model config
│── age_net.caffemodel          # Age detection weights
│── gender_deploy.prototxt      # Gender detection model config
│── gender_net.caffemodel       # Gender detection weights
│── opencv_face_detector.pbtxt  # Face detection config
│── opencv_face_detector_uint8.pb # Face detection weights
│── testvideo                   # Output video (generated after run)
```

## ⚙️ Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Aarzoo2004/IdentityScanner.git
   cd IdentityScanner
   ```

2. **Install Dependencies**

   ```bash
   pip install opencv-python numpy
   ```

3. **Run the Project**

   ```bash
   python main.py
   ```

4. **Controls**

   * A webcam window will open showing detections.
   * Press **q** to exit the program.
   * The live feed is saved as `testvideo`.

## 🎯 How It Works

1. Captures frames from webcam.
2. **Face Detection** using OpenCV’s pre-trained model.
3. Extracts features from each face using **CNN**.
4. **SVM** classifies the features into **Male/Female** and predicts the **Age Group**.
5. Displays results in real time with bounding boxes and labels.

## 📌 Notes

* Accuracy depends on lighting and camera quality.
* Modify `MODEL_MEAN_VALUES` in `main.py` for better results if needed.
* To process a video file instead of webcam:

  ```python
  cap = cv2.VideoCapture("sample_video.mp4")
  ```


