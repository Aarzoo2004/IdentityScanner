// Import libraries
const cv2 = require('cv2');
const math = require('math');
const argparse = require('argparse');

// Define file paths
const faceProto = "opencv_face_detector.pbtxt";
const faceModel = "opencv_face_detector_uint8.pb";
const ageProto = "age_deploy.prototxt";
const ageModel = "age_net.caffemodel";
const genderProto = "gender_deploy.prototxt";
const genderModel = "gender_net.caffemodel";

// Read models
const faceNet = cv2.dnn.readNet(faceModel, faceProto);
const ageNet = cv2.dnn.readNet(ageModel, ageProto);
const genderNet = cv2.dnn.readNet(genderModel, genderProto);

// Define constants
const MODEL_MEAN_VALUES = [78.4263377603, 87.7689143744, 114.895847746];
const ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'];
const genderList = ['Male', 'Female'];
const padding = 20;

// Define faceBox function
function faceBox(faceNet, frames) {
    const frameHeight = frames.shape[0];
    const frameWidth = frames.shape[1];
    const blob = cv2.dnn.blobFromImage(frames, 1.0, [300, 300], [104, 117, 123], false);
    faceNet.setInput(blob);
    const detection = faceNet.forward();
    const bboxs = [];
    for (let i = 0; i < detection.shape[2]; i++) {
        const confidence = detection[0][0][i][2];
        if (confidence > 0.7) {
            const x1 = Math.floor(detection[0][0][i][3] * frameWidth);
            const y1 = Math.floor(detection[0][0][i][4] * frameHeight);
            const x2 = Math.floor(detection[0][0][i][5] * frameWidth);
            const y2 = Math.floor(detection[0][0][i][6] * frameHeight);
            bboxs.push([x1, y1, x2, y2]);
            cv2.rectangle(frames, [x1, y1], [x2, y2], [0, 255, 0], 1);
        }
    }
    return [frames, bboxs];
}

// Define DisplayVid function
function DisplayVid() {
    const cap = cv2.VideoCapture(0, cv2.CAP_DSHOW);
    const fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D');
    const out = cv2.VideoWriter('testvideo', fourcc, 20.0, [640, 480]);
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080);
    while (true) {
        const frame = cap.read();
        const [frameFace, bboxes] = faceBox(faceNet, frame);
        const gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        out.write(frame);
        for (const bbox of bboxes) {
            const face = frame.roi(bbox[1], bbox[3], bbox[0], bbox[2]);
            const blob = cv2.dnn.blobFromImage(face, 1.0, [227, 227], MODEL_MEAN_VALUES, false);
            genderNet.setInput(blob);
            const genderPreds = genderNet.forward();
            const gender = genderList[genderPreds[0].argMax()];
            ageNet.setInput(blob);
            const agePreds = ageNet.forward();
            const age = ageList[agePreds[0].argMax()];
            const label = `${gender},${age}`;
            cv2.rectangle(frameFace, [bbox[0], bbox[1] - 30], [bbox[2], bbox[1]], [0, 255, 0], -1);
            cv2.putText(frameFace, label, [bbox[0], bbox[1] - 10], cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2, cv2.LINE_AA);
        }
        cv2.imshow("Age-Gender", frameFace);
        const k = cv2.waitKey(1);
        if (k === 113) {
            break;
        }
        if (!cap.isOpened()) {
            console.log("Could not open video device");
        }
    }
    cap.release();
    out.release();
    cv2.destroyAllWindows();
}

// Call DisplayVid function
DisplayVid();

