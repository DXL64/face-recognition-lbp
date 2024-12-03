import cv2
import time
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius = 2 , neighbors = 16 , grid_x = 9 , grid_y  = 9)
recognizer.read("face-gender-model.yml")

def face_detection(image):
  cascadePath = "haarcascade_frontalface_default.xml"
  detector = cv2.CascadeClassifier(cascadePath)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=10, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)

  return rects

if __name__ == "__main__":
  cap = cv2.VideoCapture(0)
  while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
      break
    boxes = face_detection(frame)
    for (x , y , w , h) in boxes:
        # extract the face ROI, resize it and convert
        # it into grayscale format
        faceROI = frame[y:y+h , x:x+w]
        faceROI = cv2.resize(faceROI , (47 , 62))
        faceROI = cv2.cvtColor(faceROI , cv2.COLOR_BGR2GRAY)

        (prediction , conf) = recognizer.predict(faceROI)
        
        # Show the prediction
        cv2.putText(frame , "{}".format(prediction) , (x , y - 10) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 255) , 2)
        cv2.rectangle(frame , (x , y) , (x+w , y+h) , (0 , 0 , 255) , 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame , "FPS: {:.2f}".format(fps) , (10 , 30) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 255) , 2)

    cv2.imshow("Frame" , frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      break
  cap.release()
  cv2.destroyAllWindows()