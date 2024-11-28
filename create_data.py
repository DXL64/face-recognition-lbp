import os
import shutil
import random
from PIL import Image
import numpy as np
import onnxruntime as ort
import mediapipe
import cv2
mp_face_detection = mediapipe.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Load your pre-trained gender classification model
model_path = "fa-20221226-thumb-fscore-4t.onnx"
session = ort.InferenceSession(model_path)

# Define the paths for your dataset and output folders
dataset_path = 'vietnamese-celebrity-faces'
male_folder = 'gender/male'
female_folder = 'gender/female'

# Ensure output folders exist
os.makedirs(male_folder, exist_ok=True)
os.makedirs(female_folder, exist_ok=True)

# Function to classify gender from an image
def classify_gender(img_path):
    img = cv2.imread(img_path) # Adjust target size as needed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_img = img
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype(np.float32) / 255.0
    face_img = np.transpose(face_img, (2, 0, 1))
    face_img = np.expand_dims(face_img, axis=0)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: face_img})

    gender = np.argmax(outputs[1])
    
    # Assuming the model returns 0 for male, 1 for female
    return 'male' if gender == 0 else 'female'

# Loop through each folder (label)
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    
    if os.path.isdir(folder_path):  # Ensure it's a folder
        images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('jpg', 'jpeg', 'png'))]
        if len(images) > 0:
            print(folder_path)
            # Randomly select 3 images from the folder
            selected_images = random.sample(images, min(len(images), 3))
            
            for img_path in selected_images:
                # Classify gender of the image
                gender = classify_gender(img_path)
                
                # Move image to appropriate folder
                if gender == 'male':
                    shutil.copy(img_path, male_folder)
                else:
                    shutil.copy(img_path, female_folder)

print("Image classification and sorting completed.")
