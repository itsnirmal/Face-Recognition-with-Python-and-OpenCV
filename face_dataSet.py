from cv2 import cv2
import os

#To load pre-trained frontal face data from haar cascade algorithm--opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

# Face Id for each person
face_id = input('\n enter user id end press <return> ==>  ')
name = input('enter user name ==> ')
print('[INFO] Initializing face detection, look at the camera and wait . . . ')


# Individual sampling face count
count = 0

while True:

    #Read current frame/picture
    successful_frame_read, frame = webcam.read()

    # Must convert images to grey-scale 
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))

    # Draw a rectangle around the face
    for (x, y, w, h) in face_coordinates:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        count += 1

        # Save the captured image into dataSet folder
        cv2.imwrite('dataSet/' + name + '.' + str(face_id) + '.' + str(count) + '.jpg', grayscaled_img[y:y+h, x:x+w])

         # To show in an app
        cv2.imshow('Image', frame)

    # Pause execution of progrmam
    key = cv2.waitKey(10)

    if key == 81 or key == 113:
        break
    elif count >=200: #Take 30 samples and stop execution
        break

print('\n [INFO] Exiting Program')
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")