import cv2 as cv
import math
import time

# Load face, age, and gender detection models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)

# Define model mean values, age, and gender lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-25)', '(26-30)', '(31-35)', '(36-40)', '(41-45)', '(46-50)', '(51-55)', '(56-60)', '(61-65)', '(66-70)', '(71-75)', '(76-80)', '(81-85)', '(86-90)', '(91-95)', '(96-100)']
genderList = ['Male', 'Female']
padding = 20

# Function to detect faces in a frame
def getFaceBox(frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    faceNet.setInput(blob)
    detections = faceNet.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes

# Function to perform age and gender detection on detected faces
def age_gender_detector(frame):
    frameFace, bboxes = getFaceBox(frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv.LINE_AA)
    return frameFace

# Main function to capture video from camera and perform age and gender detection
def DisplayVid():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    while (True):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Perform age and gender detection on the frame
        frame_with_detection = age_gender_detector(frame)

        # Display the frame with age and gender detection
        cv.imshow("Age-Gender Detection", frame_with_detection)

        # Break loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Call the main function to start the video capture and age-gender detection
DisplayVid()
