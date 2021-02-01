from cv2 import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataSet'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#To load pre-trained frontal face data from haar cascade algorithm--opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to get images and label data
def getImagesAndLables(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    samples = []
    ids = []

    for imagePath in imagePaths:

        
        PIL_image = Image.open(imagePath).convert('L') #convert it into grayscale
        image_numpy = np.array(PIL_image, 'uint8')

        id = int(os.path.split(imagePath)[-1].split('.')[1])
        # Detect faces
        face_coordinates = trained_face_data.detectMultiScale(image_numpy)

        # Draw a rectangle around the face
        for (x, y, w, h) in face_coordinates:
            samples.append(image_numpy[y:y+h, x:x+w])
            ids.append(id)
    
    return samples, ids

print('\n [INFO] Training faces. It will take a few seconds. Wait ...')
samples, ids = getImagesAndLables(path)
face_recognizer.train(samples, np.array(ids))

# Save model into trainedData/trainer.yml
face_recognizer.write('trainedData/trainer.yml') 

# Print number of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

print("Code Completed")
