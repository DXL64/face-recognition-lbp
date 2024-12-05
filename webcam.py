import cv2
import time
import numpy as np
import argparse

# Load the pre-trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-gender-model.yml")

def face_detection(image):
    cascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascadePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    return rects

def process_frame(frame):
    boxes = face_detection(frame)
    for (x, y, w, h) in boxes:
        # Extract the face ROI, resize it, and convert to grayscale
        faceROI = frame[y:y+h, x:x+w]
        faceROI = cv2.resize(faceROI, (47, 62))
        faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

        # Predict gender
        prediction, conf = recognizer.predict(faceROI)
        prediction = "Female" if prediction == 0 else "Male"

        # Draw predictions and bounding boxes
        cv2.putText(frame, "{}".format(prediction), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Confidence: {:.2f}".format(conf), (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return frame

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    processed_frame = process_frame(image)
    cv2.imshow("Image", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        end_time = time.time()

        # Display FPS
        fps = 1 / (end_time - start_time)
        cv2.putText(processed_frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("Video", processed_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection and gender prediction")
    parser.add_argument("--image", type=str, help="Path to the input image")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    parser.add_argument("--camera", action="store_true", help="Use webcam as input")

    args = parser.parse_args()

    if args.image:
        process_image(args.image)
    elif args.video:
        process_video(args.video)
    elif args.camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access the webcam")
        else:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame)
                end_time = time.time()

                # Display FPS
                fps = 1 / (end_time - start_time)
                cv2.putText(processed_frame, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("Webcam", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Please specify an input source (image, video, or webcam)")
