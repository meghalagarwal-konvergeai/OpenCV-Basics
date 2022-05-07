import os
import cv2 as cv
import numpy as np

# Creating a List of Folders from which the detection need to be done
people =[person for person in os.listdir(r"/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Photos/Faces")]
# Storing the path
DIR = r'/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Photos/Faces'

haar_cascade = cv.CascadeClassifier('/home/meghal/Personal/Konverge_AI/Training/OpenCV Basics/Face Detection and Recognition/haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person) # Joining the path with the names of the folder
        label = people.index(person) # Storing the index of the names from the list

        # Looping through each image of all the folders
        for img in os.listdir(path):
            img_path = os.path.join(path,img) # Joining the path with the specific image of the respective folder

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            
            # Drawing a rectangle on the face of the image
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

# Defining the inbuild Face Recognizer method of OpenCV
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)