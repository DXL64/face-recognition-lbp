from imutils import paths
import numpy as np
import imutils
import cv2
import os
from tqdm import tqdm

def face_detection(image):
  cascadePath = "haarcascade_frontalface_default.xml"
  detector = cv2.CascadeClassifier(cascadePath)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=10, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)

  return rects

def load_face_dataset(inputPath, minSamples = 15):
  # get all the image paths in the dataset folder structure and grab 
  # the name(i.e. groundtruth) of all the images and count each of them
  # and then put all of the groundtruths into a list
  imagePaths = list(paths.list_images(inputPath))

  names = [p.split(os.path.sep)[-2] for p in imagePaths]
  (names , counts) = np.unique(names , return_counts = True)
  names = names.tolist()

  faces = []
  labels = []

  print("Number of images: ", len(imagePaths))
  # loop over all of the image paths
  for imagePath in tqdm(imagePaths):
    # read the image and grab the image label
    image = cv2.imread(imagePath)
    name = imagePath.split(os.path.sep)[-2]

    # check whether the count of this specific label is
    # below our minSamples threshold or not
    if counts[names.index(name)] < minSamples:
      continue
    
    # resize it and convert it into grayscale format
    faceROI = image
    faceROI = cv2.cvtColor(faceROI , cv2.COLOR_BGR2GRAY)

    # update the faces and labels list

    faces.append(faceROI)
    labels.append(name)
  # convert the faces and labels lists into Numpy array

  faces = np.array(faces)
  labels = np.array(labels)

  return (faces , labels)

if __name__ == "__main__":
  load_face_dataset("gender_small")