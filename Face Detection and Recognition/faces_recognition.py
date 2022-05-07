import os
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Face Detection and Recognition/haar_face.xml')

people =[person for person in os.listdir(r"/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Photos/Faces")]
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/face_trained.yml')

img = cv.imread(r'/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Photos/val/index6.jpeg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    # Fetching the Predicted name and the Accuracy of the Predictions
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # Adding a Name and a rectangle on the image 
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)