import cv2
import numpy as np
import face_recognition
import os
import mediapipe
import onnxruntime as ort

mp_face_detection = mediapipe.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Initialize the webcam
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("../6.mp4")

model_path = "fa-20221226-thumb-fscore-4t.onnx"
session = ort.InferenceSession(model_path)

ageList = ['(0-12)', '(13-17)', '(18-24)', '(25-34)', '(35-44)', '(45-54)', '(55-64)', '(65-100)']
genderList = ['Female', 'Male']
maskList = ['No', 'Yes']
glassList = ['No', 'Yes']

def load_known_faces(directory):    
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use the filename without extension as the name
    
    return known_face_encodings, known_face_names

def main():
    known_face_encodings = []
    known_face_names = []
    known_face_encodings, known_face_names = load_known_faces("./database")
    try:
        while True:
            # Read a frame
            ret, frame = cap.read()

            if ret:
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Find all the faces and face encodings in the current frame of video
                face_locations = []
                frame_height, frame_width, _ = frame.shape
                # face_locations = face_recognition.face_locations(frame)
                if results.detections is not None:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        # Convert relative coordinates to pixel coordinates
                        x = int(bbox.xmin * frame_width)
                        y = int(bbox.ymin * frame_height)
                        width = int(bbox.width * frame_width)
                        height = int(bbox.height * frame_height)

                        top = y
                        left = x
                        bottom = y + height
                        right = x + width

                        face_locations.append((top, right, bottom, left))

                face_encodings = face_recognition.face_encodings(frame, face_locations, model="large")

                face_names = []
                for face_encoding in face_encodings:
                    name = "Khach"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < 0.5: #threshold
                            name = known_face_names[best_match_index]

                    face_names.append(name)

                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    face_img = frame[top:bottom, left:right]
                    face_img = cv2.resize(face_img, (128, 128))
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_img = face_img.astype(np.float32) / 255.0
                    face_img = np.transpose(face_img, (2, 0, 1))
                    face_img = np.expand_dims(face_img, axis=0)
                    
                    input_name = session.get_inputs()[0].name
                    outputs = session.run(None, {input_name: face_img})

                    age = ageList[np.argmax(outputs[0])]
                    gender = genderList[np.argmax(outputs[1])]
                    glass = glassList[np.argmax(outputs[2])]
                    mask = maskList[np.argmax(outputs[3])]

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

                    cv2.putText(frame, f"Age: {age}", (left + 75, top), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
                    cv2.putText(frame, f"Gender: {gender}", (left + 75, top + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
                    cv2.putText(frame, f"Glass: {glass}", (left + 75, top + 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
                    cv2.putText(frame, f"Mask: {mask}", (left + 75, top + 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 1)
                cv2.imshow("Webcam Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
