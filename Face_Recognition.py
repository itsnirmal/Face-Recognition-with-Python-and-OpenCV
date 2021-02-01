from cv2 import cv2
import numpy as np
import os
#from face_dataSet import name

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainedData/trainer.yml')
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0
# names related to IDs
names = [ 'None','Nirmal' , 'Raj', 'Lochan', 'Gita', 'Ramesh']

webcam = cv2.VideoCapture(0)

while True:

    #Read current frame/picture
    successful_frame_read, frame = webcam.read()

    # Must convert images to grey-scale 
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        id, confidence = face_recognizer.predict(grayscaled_img[y:y+h,x:x+w])
        
        if (confidence < 100) :
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
             id = "Imposter" # \(>_<)/
             confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    frame, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255, 255, 255), 
                    2
                   )
        cv2.putText(
                    frame, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    1
                   )  
    # To show in an app
    cv2.imshow('Face Recognizer', frame)
    # Pause execution of progrmam
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()


print("Code Completed")





